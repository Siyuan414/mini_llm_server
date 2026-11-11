/**
 * paged_attention.cuh — decode attention over a paged KV cache (S2.8).
 *
 * Shapes
 *   q           [B, Hq, D]                 one query token per sequence
 *   k_pool      [num_blocks, Hkv, BS, D]   physical block storage
 *   v_pool      [num_blocks, Hkv, BS, D]
 *   block_table [B, max_blocks]            logical -> physical, -1 = unused
 *   lens        [B]                        real cached length per sequence
 *   out         [B, Hq, D]
 *
 * One thread block per (sequence, query head): grid = (Hq, B).
 * Dynamic shared memory:  [ scores (len) | reduce (kThreads) | table (max_blocks) ]
 *
 * ── Why Phase 3 iterates block-major ─────────────────────────────────────────
 * The obvious version loops `for t in [0,len)` and derives the address from t:
 *
 *     logical = t / BS;  offset = t % BS;  physical = table[logical];
 *
 * In Phase 1 that is fine — threads stride over t, so each token's division
 * happens once (~len per block). In Phase 3 every one of the D threads walks
 * EVERY token, so the same divisions repeat D times: ~D*len = 3200 integer
 * divisions per block at D=64, len=50. Integer division costs ~20 cycles, and
 * benchmarking showed this made paged attention SLOWER than the fixed-slot
 * kernel it replaces, despite scanning 10x fewer tokens.
 *
 * Looping block-major instead — outer over logical blocks, inner over offsets —
 * removes the division entirely and reads the table once per BS tokens.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>

namespace paged_attn {

constexpr int kThreads = 64;

/// Block-wide reduction over `val`. All threads must reach this.
template <typename Op>
__device__ inline float block_reduce(float val, float* smem, Op op) {
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = op(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    const float out = smem[0];
    __syncthreads();
    return out;
}

__global__ void paged_attention_decode(
        const __half* __restrict__ q,            // [B, Hq, D]
        const __half* __restrict__ k_pool,       // [NB, Hkv, BS, D]
        const __half* __restrict__ v_pool,       // [NB, Hkv, BS, D]
        const int*    __restrict__ block_table,  // [B, max_blocks]
        const int*    __restrict__ lens,         // [B]
        __half*       __restrict__ out,          // [B, Hq, D]
        int Hq, int Hkv, int D, int BS,
        int max_blocks, float scale) {

    // ── Phase 0: identity ───────────────────────────────────────────────────
    const int h   = blockIdx.x;                  // query head
    const int b   = blockIdx.y;                  // sequence
    const int tid = threadIdx.x;
    const int len = lens[b];

    if (len <= 0) {                              // empty slot: never touch KV
        for (int d = tid; d < D; d += blockDim.x)
            out[((size_t)b * Hq + h) * D + d] = __float2half(0.0f);
        return;
    }

    const int kv_head = h / (Hq / Hkv);          // GQA: 4 q heads share a kv head
    const __half* qv  = q + ((size_t)b * Hq + h) * D;

    // [ scores(len) | reduce(kThreads) | table(max_blocks) ] — all 4-byte types
    extern __shared__ float smem[];
    float* scores    = smem;
    float* reduce    = scores + len;
    int*   table_row = (int*)(reduce + kThreads);

    // Cache this sequence's block table in shared memory: read once from global,
    // then Phases 1 and 3 hit shared instead.
    for (int i = tid; i < max_blocks; i += blockDim.x)
        table_row[i] = block_table[(size_t)b * max_blocks + i];
    __syncthreads();

    const int n_lb = (len + BS - 1) / BS;        // logical blocks in use

    // ── Phase 1: scores[t] = dot(q, K[t]) * scale ───────────────────────────
    // Threads stride over tokens, so each division happens once per token.
    for (int t = tid; t < len; t += blockDim.x) {
        const int logical  = t / BS;
        const int offset   = t % BS;
        const int physical = table_row[logical];
        if (physical < 0) { scores[t] = -FLT_MAX; continue; }   // fail loud

        const __half* kv =
            k_pool + (((size_t)physical * Hkv + kv_head) * BS + offset) * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d)
            acc += __half2float(qv[d]) * __half2float(kv[d]);
        scores[t] = acc * scale;
    }
    __syncthreads();

    // ── Phase 2: softmax ────────────────────────────────────────────────────
    // Subtracting the max is numerical stability (expf overflows fp32 near 88).
    // It needs a BLOCK-wide reduction because tokens are spread across threads.
    float local_max = -FLT_MAX;
    for (int t = tid; t < len; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    const float m = block_reduce(local_max, reduce,
        [] __device__ (float x, float y) { return fmaxf(x, y); });

    float local_sum = 0.0f;
    for (int t = tid; t < len; t += blockDim.x) {
        const float e = __expf(scores[t] - m);
        scores[t]  = e;
        local_sum += e;
    }
    __syncthreads();
    const float l = block_reduce(local_sum, reduce,
        [] __device__ (float x, float y) { return x + y; });
    const float inv_l = 1.0f / l;

    // ── Phase 3: out[d] = Σ p[t] · V[t][d] ──────────────────────────────────
    // Thread d owns output dimension d, so the accumulator stays in a register
    // and no final reduction is needed. BLOCK-MAJOR: one table read and zero
    // divisions per BS tokens (see the header note).
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int lb = 0; lb < n_lb; ++lb) {
            const int physical = table_row[lb];
            if (physical < 0) continue;
            const int t0 = lb * BS;
            const int n  = (len - t0 < BS) ? (len - t0) : BS;   // last block partial
            const __half* base =
                v_pool + ((size_t)physical * Hkv + kv_head) * BS * D;
            for (int o = 0; o < n; ++o)
                acc += scores[t0 + o] * __half2float(base[(size_t)o * D + d]);
        }
        out[((size_t)b * Hq + h) * D + d] = __float2half(acc * inv_l);
    }
}

}  // namespace paged_attn

/// max_len is the largest value in `lens` (host-side), used to size shared mem.
inline void launch_paged_attention_decode(
        const __half* q, const __half* k_pool, const __half* v_pool,
        const int* block_table, const int* lens, __half* out,
        int B, int Hq, int Hkv, int D, int BS,
        int max_blocks, int max_len, float scale, cudaStream_t stream) {
    if (Hq % Hkv != 0) throw std::runtime_error("Hq must be a multiple of Hkv");
    const dim3 grid(Hq, B);
    // scores + reduce (floats) + the cached block table (ints)
    const size_t smem = (size_t)(max_len + paged_attn::kThreads) * sizeof(float)
                      + (size_t)max_blocks * sizeof(int);
    paged_attn::paged_attention_decode<<<grid, paged_attn::kThreads, smem, stream>>>(
        q, k_pool, v_pool, block_table, lens, out,
        Hq, Hkv, D, BS, max_blocks, scale);
}
