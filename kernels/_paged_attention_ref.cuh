/**
 * paged_attention.cuh — decode attention over a paged KV cache (Stage 2B, S2.8).
 *
 * ── The one thing this kernel exists to do ───────────────────────────────────
 * Contiguous attention finds token t at   base + t * D.
 * Paged attention finds it at             pool + block_table[logical] * BLOCK
 *                                              + offset * D
 * where logical = t / block_size and offset = t % block_size.
 *
 * That indirection is why TRT's compiled attention cannot be reused: `base +
 * t*D` is machine code inside the engine. Everything else here is ordinary
 * attention.
 *
 * ── Shapes ───────────────────────────────────────────────────────────────────
 *   q           [B, Hq, D]                    one query token per sequence
 *   k_pool      [num_blocks, Hkv, BS, D]      physical block storage
 *   v_pool      [num_blocks, Hkv, BS, D]
 *   block_table [B, max_blocks]               logical -> physical, -1 = unused
 *   lens        [B]                           real cached length per sequence
 *   out         [B, Hq, D]
 *
 * ── Why not online softmax ───────────────────────────────────────────────────
 * Flash attention streams softmax to avoid materializing an N x N score matrix.
 * In decode the query is ONE token, so the matrix is 1 x N — 512 floats, 2 KB
 * of shared memory. Three straightforward passes are easier to verify and cost
 * nothing here. (Prefill attention, with N queries, is where online softmax
 * earns its complexity.)
 *
 * ── GQA ──────────────────────────────────────────────────────────────────────
 * LLaMA-3.2-1B has 32 query heads and 8 KV heads, so 4 query heads share each
 * KV head: kv_head = q_head / (Hq / Hkv).
 *
 * ── Parallelisation ──────────────────────────────────────────────────────────
 * One thread block per (sequence, query head): grid = (Hq, B). Each block owns
 * one D-vector of output, so no cross-block coordination is needed.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>

namespace paged_attn {

constexpr int kThreads = 128;

/// Block-wide reduction. `op` is a binary functor over float.
template <typename Op>
__device__ inline float block_reduce(float val, float* smem, Op op, float init) {
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

/**
 * scores_smem must hold max_len floats; the reduction reuses the tail.
 * Dynamic shared memory layout: [ scores (max_len) | reduce (kThreads) ]
 */
__global__ void paged_attention_decode(
        const __half* __restrict__ q,            // [B, Hq, D]
        const __half* __restrict__ k_pool,       // [NB, Hkv, BS, D]
        const __half* __restrict__ v_pool,       // [NB, Hkv, BS, D]
        const int*    __restrict__ block_table,  // [B, max_blocks]
        const int*    __restrict__ lens,         // [B]
        __half*       __restrict__ out,          // [B, Hq, D]
        int Hq, int Hkv, int D, int BS,
        int max_blocks, float scale) {

    const int h = blockIdx.x;      // query head
    const int b = blockIdx.y;      // sequence
    const int tid = threadIdx.x;

    const int len = lens[b];
    if (len <= 0) {                // empty slot: emit zeros, do not read KV
        for (int d = tid; d < D; d += blockDim.x)
            out[((size_t)b * Hq + h) * D + d] = __float2half(0.f);
        return;
    }

    const int kv_head = h / (Hq / Hkv);                 // GQA mapping
    const __half* qv  = q + ((size_t)b * Hq + h) * D;

    extern __shared__ float smem[];
    float* scores = smem;                               // [len]
    float* reduce = smem + len;                         // [kThreads]

    // ── Phase 1: scores[t] = dot(q, K[t]) * scale ───────────────────────────
    // Each thread strides over tokens and does the full D-dim dot product.
    for (int t = tid; t < len; t += blockDim.x) {
        const int logical  = t / BS;
        const int offset   = t % BS;
        const int physical = block_table[(size_t)b * max_blocks + logical];
        // A -1 here means the allocator and the kernel disagree about how many
        // blocks this sequence owns — fail loudly rather than read block 0.
        if (physical < 0) { scores[t] = -FLT_MAX; continue; }

        const __half* kv =
            k_pool + (((size_t)physical * Hkv + kv_head) * BS + offset) * D;

        float acc = 0.f;
        for (int d = 0; d < D; ++d)
            acc += __half2float(qv[d]) * __half2float(kv[d]);
        scores[t] = acc * scale;
    }
    __syncthreads();

    // ── Phase 2: softmax over [0, len) ──────────────────────────────────────
    float local_max = -FLT_MAX;
    for (int t = tid; t < len; t += blockDim.x) local_max = fmaxf(local_max, scores[t]);
    const float m = block_reduce(local_max, reduce,
                                 [] __device__ (float a, float c) { return fmaxf(a, c); },
                                 -FLT_MAX);

    float local_sum = 0.f;
    for (int t = tid; t < len; t += blockDim.x) {
        const float e = __expf(scores[t] - m);          // stable: max subtracted
        scores[t] = e;
        local_sum += e;
    }
    __syncthreads();
    const float l = block_reduce(local_sum, reduce,
                                 [] __device__ (float a, float c) { return a + c; }, 0.f);
    const float inv_l = 1.f / l;

    // ── Phase 3: out = Σ p[t] · V[t] ────────────────────────────────────────
    // Thread d owns output dimension d and walks the whole sequence, so the
    // accumulator stays in a register and no further reduction is needed.
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < len; ++t) {
            const int logical  = t / BS;
            const int offset   = t % BS;
            const int physical = block_table[(size_t)b * max_blocks + logical];
            if (physical < 0) continue;
            const __half* vv =
                v_pool + (((size_t)physical * Hkv + kv_head) * BS + offset) * D;
            acc += scores[t] * __half2float(vv[d]);
        }
        out[((size_t)b * Hq + h) * D + d] = __float2half(acc * inv_l);
    }
}

}  // namespace paged_attn

/// Launcher. max_len is the largest value in `lens` (host-side), used to size
/// dynamic shared memory.
inline void launch_paged_attention_decode(
        const __half* q, const __half* k_pool, const __half* v_pool,
        const int* block_table, const int* lens, __half* out,
        int B, int Hq, int Hkv, int D, int BS,
        int max_blocks, int max_len, float scale, cudaStream_t stream) {
    if (Hq % Hkv != 0) throw std::runtime_error("Hq must be a multiple of Hkv");

    const dim3 grid(Hq, B);
    const size_t smem = (size_t)(max_len + paged_attn::kThreads) * sizeof(float);
    paged_attn::paged_attention_decode<<<grid, paged_attn::kThreads, smem, stream>>>(
        q, k_pool, v_pool, block_table, lens, out,
        Hq, Hkv, D, BS, max_blocks, scale);
}
