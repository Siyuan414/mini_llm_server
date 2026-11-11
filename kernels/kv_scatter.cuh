/**
 * kv_scatter.cuh — copy the newly-generated KV entry into our own cache.
 *
 * ── The problem ──────────────────────────────────────────────────────────────
 * After a decode step TRT gives us `present`, which is the whole cache with one
 * new token appended at the end (index `past`). We keep our own `cache` with a
 * FIXED stride of max_seq, because a fixed stride is what lets slots be
 * independent — a new request can be written into row i without disturbing
 * anyone else.
 *
 * So each step we move one token per (slot, head) out of TRT's buffer into ours:
 *
 *     present[b, h, past,    :]   ──►   cache[b, h, lens[b], :]
 *              ^^^^                              ^^^^^^^
 *         same for every slot              DIFFERENT per slot
 *
 * That mismatch is why this is a kernel and not a cudaMemcpy. A memcpy moves one
 * contiguous run to one destination; here every slot has its own destination
 * index. This is a *scatter*: one source pattern, many indexed destinations.
 *
 * (The same indirection, one level deeper, is what paged attention does in Stage
 * 2B — there the destination comes from a block table instead of a length array.)
 *
 * ── Why not a loop of memcpys ────────────────────────────────────────────────
 * B x H x layers x 2 small copies per step — ~1000 launches at real sizes, each
 * with launch overhead bigger than the copy itself. One kernel does all of it.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kv_kernels {

/**
 * present : [B, H, past+1, D]  — TRT output; new token sits at index `past`
 * cache   : [B, H, max_seq, D] — ours; new token goes to index lens[b]
 * lens    : [B] on device      — each slot's CURRENT length (before this token)
 *
 * One thread per element. Total work = B * H * D.
 */
__global__ void scatter_new_kv(const __half* __restrict__ present,
                               __half* __restrict__ cache,
                               const int* __restrict__ lens,
                               int B, int H, int D,
                               int past, int max_seq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H * D;
    if (i >= total) return;              // last block is usually partial

    // ── Unflatten: which (slot, head, element) am I? ─────────────────────────
    // Exact inverse of the offset formula: innermost dim varies fastest, so
    // modulo recovers it; divide past it to reach the next dim out.
    const int d = i % D;
    const int h = (i / D) % H;
    const int b = i / (D * H);

    const int t = lens[b];               // this slot's own write position
    if (t >= max_seq) return;            // slot is full — scheduler must evict

    // ── Flatten: each index times the product of all sizes to its right ──────
    const size_t src = (size_t)b * H * (past + 1) * D
                     + (size_t)h * (past + 1) * D
                     + (size_t)past * D
                     + d;

    const size_t dst = (size_t)b * H * max_seq * D
                     + (size_t)h * max_seq * D
                     + (size_t)t * D
                     + d;

    cache[dst] = present[src];
}

}  // namespace kv_kernels

/**
 * Launch helper. Call once per layer for K and once for V.
 *
 * 256 threads/block is a reasonable default: enough to fill a warp scheduler,
 * small enough that a partial last block wastes little. This kernel is purely
 * memory-bound and tiny (B*H*D = 2048 elements at B=4), so occupancy tuning is
 * not worth it — the win over the memcpy version is launch count, not bandwidth.
 */
inline void launch_scatter_new_kv(const __half* present, __half* cache,
                                  const int* lens,
                                  int B, int H, int D,
                                  int past, int max_seq,
                                  cudaStream_t stream) {
    const int total   = B * H * D;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;   // round UP
    kv_kernels::scatter_new_kv<<<blocks, threads, 0, stream>>>(
        present, cache, lens, B, H, D, past, max_seq);
}
