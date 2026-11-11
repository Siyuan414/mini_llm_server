/**
 * paged_attention_varlen.cuh — attention for q_len >= 1 over a paged KV cache (S3.1).
 *
 * Generalizes paged_attention.cuh from "one query token per sequence" to "any
 * number of query tokens per sequence, different per sequence". That single
 * change covers three cases with no branching:
 *
 *     decode          1 query token  per sequence   (the old kernel)
 *     prefill chunk   N query tokens per sequence   (chunked prefill, S3)
 *     verification    K query tokens per sequence   (speculative decoding, S4)
 *
 * ── Flat query layout ────────────────────────────────────────────────────────
 * Query tokens from all sequences are concatenated:
 *
 *     seq 0 prefilling 236 tokens ┐
 *     seq 1 prefilling  20 tokens ├─ q = [total_q, Hq, D]
 *     seq 2 decoding     1 token  ┘   total_q = 257
 *
 * blockIdx.y is a GLOBAL query index g in [0, total_q). It alone cannot say
 * which sequence g belongs to, so two host-built arrays carry that:
 *
 *     q_seq[g]   sequence index -> which block_table row to read
 *     q_pos[g]   ABSOLUTE position in that sequence
 *
 * ── The mask is a loop bound, not a branch ───────────────────────────────────
 * Causality says query at absolute position p sees keys 0..p. So this block
 * scans exactly q_pos[g] + 1 keys and never computes a score it would discard.
 *
 * Note what this removes: `lens[b]` is gone. It existed only to say how many
 * keys to scan, and q_pos[g] + 1 says it per query token instead of per
 * sequence — strictly more precise. Decode is the case q_pos[g] == len - 1,
 * which scans the whole cache, exactly as before.
 *
 * ── Ordering requirement (this is the trap) ──────────────────────────────────
 * Query token p attends to key p — ITSELF. So write_kv_paged must have written
 * K/V for every query token in this batch BEFORE this kernel launches. Getting
 * this off by one produces fluent-but-wrong output, the same failure signature
 * as the S2.7 `lens` bug (rel err 0.129711).
 *
 * ── Host side ────────────────────────────────────────────────────────────────
 * Build the two arrays while building the batch:
 *
 *     for each sequence s in running:
 *         for i in [0, s.chunk_len):
 *             q_seq.push_back(slot_of(s));
 *             q_pos.push_back(s.len + i);      // absolute
 *
 * max_scan = 1 + max(q_pos) sizes the score array in shared memory. Each block
 * partitions with its OWN n_keys (shared memory is per block, so blocks need
 * not agree); max_scan only has to be an upper bound for the reservation.
 */

#pragma once

#include "paged_attention.cuh"   // reuse paged_attn::block_reduce, kThreads

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>

namespace paged_attn_varlen {

using paged_attn::kThreads;
using paged_attn::block_reduce;

__global__ void paged_attention_varlen(
        const __half* __restrict__ q,            // [total_q, Hq, D]
        const __half* __restrict__ k_pool,       // [NB, Hkv, BS, D]
        const __half* __restrict__ v_pool,       // [NB, Hkv, BS, D]
        const int*    __restrict__ block_table,  // [B, max_blocks]
        const int*    __restrict__ q_seq,        // [total_q] -> sequence
        const int*    __restrict__ q_pos,        // [total_q] -> absolute position
        __half*       __restrict__ out,          // [total_q, Hq, D]
        int Hq, int Hkv, int D, int BS,
        int max_blocks, float scale) {

    // ── Phase 0: identity ───────────────────────────────────────────────────
    const int h   = blockIdx.x;                  // query head
    const int g   = blockIdx.y;                  // global query token
    const int tid = threadIdx.x;
    const int b   = q_seq[g];                    // which sequence g belongs to
    const int p   = q_pos[g];                    // absolute position of g
    const int n_keys = p + 1;                    // the causal mask, as a count
    const int kv_head = h / (Hq / Hkv);          // GQA: 4 q heads share a kv head
    const __half* qv  = q + ((size_t)g * Hq + h) * D;   // indexed by g, not b

    // [ scores(n_keys) | reduce(kThreads) | table(max_blocks) ] — all 4-byte.
    // Shared memory is PER BLOCK, so partitioning by this block's own n_keys is
    // fine — blocks never read each other's. n_keys <= max_scan, and the host
    // reserved max_scan + kThreads floats plus max_blocks ints, so every layout
    // this produces fits. Sizing by n_keys just leaves the slack at the end.
    extern __shared__ float smem[];
    float* scores    = smem;                      // [n_keys]
    float* reduce    = scores + n_keys;           // [kThreads]
    int*   table_row = (int*)(reduce + kThreads); // [max_blocks]

    // Cache this sequence's block table row: one global read per block instead
    // of one per token (see the decode kernel's note).
    for (int i = tid; i < max_blocks; i += blockDim.x)
        table_row[i] = block_table[(size_t)b * max_blocks + i];
    __syncthreads();

    const int n_lb = (n_keys + BS - 1) / BS;     // logical blocks this query spans

    // ── Phase 1: scores[t] = dot(q, K[t]) * scale, for t in [0, n_keys) ─────
    // Same as the decode kernel, with `len` replaced by `n_keys`. Threads
    // stride over t so the integer division happens once per token.
    for (int i = tid; i < n_keys; i += blockDim.x) {
        const int logical  = i / BS;
        const int offset   = i % BS;
        const int physical = table_row[logical];
        if (physical < 0) { scores[i] = -FLT_MAX; continue; }   // fail loud

        const __half* kv =
            k_pool + (((size_t)physical * Hkv + kv_head) * BS + offset) * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d)
            acc += __half2float(qv[d]) * __half2float(kv[d]);
        scores[i] = acc * scale;
    }
    __syncthreads();

    // ── Phase 2: softmax over scores[0, n_keys) ─────────────────────────────
    // Unchanged from decode — block_reduce still applies, because the reduction
    // is still over ONE query's score vector. Each query token gets its own
    // block, so nothing here has to become a 2-D reduction.
    float local_max = -FLT_MAX;
    for (int t = tid; t < n_keys; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    const float m = block_reduce(local_max, reduce,
        [] __device__ (float x, float y) { return fmaxf(x, y); });

    float local_sum = 0.0f;
    for (int t = tid; t < n_keys; t += blockDim.x) {
        const float e = __expf(scores[t] - m);
        scores[t]  = e;
        local_sum += e;
    }
    __syncthreads();
    const float l = block_reduce(local_sum, reduce,
        [] __device__ (float x, float y) { return x + y; });
    const float inv_l = 1.0f / l;
 
    // ── Phase 3: out[d] = Σ p[t] · V[t][d] ──────────────────────────────────
    // Block-major over logical blocks, as in decode, to avoid per-token integer
    // division. One subtlety: n_keys may end PART WAY into a logical block, so
    // the last block's token count is (n_keys - t0), not necessarily BS.
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int lb = 0; lb < n_lb; ++lb) {
            const int physical = table_row[lb];
            if (physical < 0) continue;
            const int t0 = lb * BS;
            const int n  = (n_keys - t0 < BS) ? (n_keys - t0) : BS;   // last block partial
            const __half* base =
                v_pool + ((size_t)physical * Hkv + kv_head) * BS * D;
            for (int o = 0; o < n; ++o)
                acc += scores[t0 + o] * __half2float(base[(size_t)o * D + d]);
        }
        out[((size_t)g * Hq + h) * D + d] = __float2half(acc * inv_l);
    }
}

}  // namespace paged_attn_varlen

/// max_scan = 1 + max(q_pos) over the batch — sizes the score array.
inline void launch_paged_attention_varlen(
        const __half* q, const __half* k_pool, const __half* v_pool,
        const int* block_table, const int* q_seq, const int* q_pos, __half* out,
        int total_q, int Hq, int Hkv, int D, int BS,
        int max_blocks, int max_scan, float scale, cudaStream_t stream) {
    if (Hq % Hkv != 0) throw std::runtime_error("Hq must be a multiple of Hkv");

    const dim3 grid(Hq, total_q);          // one block per (query token, head)
    const size_t smem =
          (size_t)(max_scan + paged_attn_varlen::kThreads) * sizeof(float)
        + (size_t)max_blocks * sizeof(int);

    // 48 KB is the default per-block limit; past that needs an opt-in via
    // cudaFuncSetAttribute(MaxDynamicSharedMemorySize). max_scan = 2048 gives
    // ~8 KB + table, so this is fine at current max_seq — but it caps how long
    // a sequence can get before the kernel must be restructured.
    paged_attn_varlen::paged_attention_varlen
        <<<grid, paged_attn_varlen::kThreads, smem, stream>>>(
            q, k_pool, v_pool, block_table, q_seq, q_pos, out,
            Hq, Hkv, D, BS, max_blocks, scale);
}
