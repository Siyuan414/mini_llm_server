/**
 * decode_layer.h — one LLaMA decoder layer, decode step (S2.7).
 *
 * TensorRT is not involved here. Seven cuBLAS GEMMs plus the kernels in
 * layer_kernels.cuh and paged_attention.cuh.
 *
 * ── The layer ────────────────────────────────────────────────────────────────
 *   h = rmsnorm(x, w_in)
 *   q = Wq·h ; k = Wk·h ; v = Wv·h
 *   rope(q, pos) ; rope(k, pos)
 *   write k, v into the paged cache at lens[b]
 *   a = paged_attention(q, cache)
 *   x += Wo·a
 *
 *   h = rmsnorm(x, w_post)
 *   x += Wdown · silu_mul(Wgate·h, Wup·h)
 *
 * ── Every GEMM has the same shape ────────────────────────────────────────────
 * Weights were exported as [in, out] row-major, which cuBLAS reads as [out, in]
 * column-major. So all seven are N/N with lda = out:
 *
 *   cublasHgemm(N, N, m=out, n=B, k=in, W, out, x, in, y, out)
 *
 *   q_proj      m=2048  k=2048        gate/up   m=8192  k=2048
 *   k_proj      m= 512  k=2048        down      m=2048  k=8192
 *   v_proj      m= 512  k=2048        o_proj    m=2048  k=2048
 *
 * k/v are m=512 not 2048 because GQA gives them 8 heads, not 32.
 */

#pragma once

#include "model_config.h"
#include "weights.h"
#include "layer_kernels.cuh"
#include "paged_attention.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

// ── Scratch buffers, allocated once ──────────────────────────────────────────
// Sized for max_batch tokens (decode: one token per sequence).
struct DecodeScratch {
    __half* h      = nullptr;   // [B, hidden]     normalized input
    __half* q      = nullptr;   // [B, Hq*D]
    __half* k      = nullptr;   // [B, Hkv*D]
    __half* v      = nullptr;   // [B, Hkv*D]
    __half* attn   = nullptr;   // [B, Hq*D]       attention output
    __half* proj   = nullptr;   // [B, hidden]     o_proj / down_proj result
    __half* gate   = nullptr;   // [B, inter]
    __half* up     = nullptr;   // [B, inter]
    __half* act    = nullptr;   // [B, inter]      silu(gate)*up

    // Gathered hidden states for the sequences that need logits this step.
    // Only a prefill chunk's LAST query token predicts anything, so lm_head runs
    // over n_seqs rows, not max_batch. At vocab 128256 that is the difference
    // between a 512-row GEMM (262 MB of logits) and a 64-row one.
    __half* h_last = nullptr;   // [max_seqs, hidden]

    /// `max_batch` is a ROW count: max_batch tokens for decode, or
    /// max_batched_tokens once chunked prefill is wired up. `max_seqs` sizes
    /// h_last; 0 means "same as max_batch".
    void alloc(const ModelConfig& cfg, int max_batch, int max_seqs = 0) {
        if (max_seqs <= 0) max_seqs = max_batch;
        cudaMalloc((void**)&h_last, (size_t)max_seqs * cfg.hidden_dim * sizeof(__half));
        cudaMalloc((void**)&h,    max_batch * cfg.hidden_dim * sizeof(__half));
        cudaMalloc((void**)&q,    max_batch * cfg.num_q_heads * cfg.head_dim * sizeof(__half));
        cudaMalloc((void**)&k,    max_batch * cfg.num_kv_heads * cfg.head_dim * sizeof(__half));
        cudaMalloc((void**)&v,    max_batch * cfg.num_kv_heads * cfg.head_dim * sizeof(__half));
        cudaMalloc((void**)&attn, max_batch * cfg.num_q_heads * cfg.head_dim * sizeof(__half));                                 
        cudaMalloc((void**)&proj, max_batch * cfg.hidden_dim * sizeof(__half));
        cudaMalloc((void**)&gate, max_batch * cfg.inter_dim * sizeof(__half));
        cudaMalloc((void**)&up,   max_batch * cfg.inter_dim * sizeof(__half));
        cudaMalloc((void**)&act,  max_batch * cfg.inter_dim * sizeof(__half));
    }
    void free() {
        cudaFree(h_last);
        cudaFree(h);
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(attn);
        cudaFree(proj);
        cudaFree(gate);
        cudaFree(up);
        cudaFree(act);
    }
};

// ── Write this step's K/V into the paged pool ────────────────────────────────
// For each (sequence, kv_head): store k[b, kvh, :] at
//   pool[ block_table[b][ lens[b]/BS ] ][ kvh ][ lens[b]%BS ][ : ]
//
// Same indirection as the attention kernel, in the opposite direction. Note it
// uses lens[b] BEFORE the increment — the new token goes at the current length.
//

__global__ void write_kv_paged(const __half* __restrict__ k,   // [B, Hkv, D]
                               const __half* __restrict__ v,   // [B, Hkv, D]
                               __half* __restrict__ k_pool,
                               __half* __restrict__ v_pool,
                               const int* __restrict__ block_table,
                               const int* __restrict__ lens,
                               int Hkv, int D, int BS, int max_blocks) {
    const int b = blockIdx.x / Hkv;
    const int kvh = blockIdx.x % Hkv;
    const int pos = lens[b] - 1;  // the new token is at the current length
    const int logical = pos / BS;
    const int offset = pos % BS;
    const int physical = block_table[b * max_blocks + logical];
    if (physical < 0) return;  // allocator and kernel disagree
    const size_t pool_idx = ((size_t)physical * Hkv + kvh) * BS * D + (size_t)offset * D;
    const size_t src_idx  = ((size_t)b * Hkv + kvh) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        k_pool[pool_idx + i] = k[src_idx + i];
        v_pool[pool_idx + i] = v[src_idx + i];  
    }

}

inline void launch_write_kv_paged(const __half* k, const __half* v,
                                  __half* k_pool, __half* v_pool,
                                  const int* block_table, const int* lens,
                                  int B, int Hkv, int D, int BS,
                                  int max_blocks, cudaStream_t s) {
    write_kv_paged<<<B * Hkv, 256, 0, s>>>(k, v, k_pool, v_pool,
                                           block_table, lens,
                                           Hkv, D, BS, max_blocks);
}

// ── GEMM helper: y[out, B] = W[out, in] · x[in, B] ───────────────────────────
inline void gemm(cublasHandle_t blas, const __half* W, const __half* x,
                 __half* y, int out, int in, int B) {
    const __half one = __float2half(1.f), zero = __float2half(0.f);
    const cublasStatus_t st = cublasHgemm(
        blas, CUBLAS_OP_N, CUBLAS_OP_N,
        out, B, in,
        &one,  W, out,      // lda = out: file is [in,out] row-major
               x, in,       // ldb = in
        &zero, y, out);     // ldc = out
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasHgemm failed");
}

// ── One decoder layer, in place on x ─────────────────────────────────────────
inline void forward_layer(const ModelConfig& cfg, const LayerWeights& w,
                          cublasHandle_t blas, DecodeScratch& s,
                          __half* x,                    // [B, hidden] in/out
                          __half* k_pool, __half* v_pool,
                          const int* d_block_table, const int* d_lens,
                          const int* d_positions,
                          int B, int block_size, int max_blocks, int max_len,
                          cudaStream_t stream) {
    const int H  = cfg.hidden_dim;
    const int QD = cfg.num_q_heads  * cfg.head_dim;
    const int KD = cfg.num_kv_heads * cfg.head_dim;
    const int I  = cfg.inter_dim;

    // ── Attention block ──────────────────────────────────────────────────────
    // TODO
    //  1. launch_rmsnorm(x, w.input_norm, s.h, B, H, cfg.rms_eps, stream)
    //  2. gemm(blas, w.q_proj, s.h, s.q, QD, H, B)   and k, v with KD
    //  3. launch_rope(s.q, d_positions, B, cfg.num_q_heads,  cfg.head_dim, ...)
    //     launch_rope(s.k, d_positions, B, cfg.num_kv_heads, cfg.head_dim, ...)
    //  4. launch_write_kv_paged(...)   ← BEFORE attention: the new token must
    //                                    be visible to itself
    //  5. launch_paged_attention_decode(s.q, k_pool, v_pool, ..., s.attn, ...)
    //     scale = 1/sqrt(head_dim)
    //  6. gemm(blas, w.o_proj, s.attn, s.proj, H, QD, B)
    //     launch_residual_add(x, s.proj, B*H, stream)

    launch_rmsnorm(x, w.input_norm, s.h, B, H, cfg.rms_eps, stream);
    gemm(blas, w.q_proj, s.h, s.q, QD, H, B);
    gemm(blas, w.k_proj, s.h, s.k, KD, H, B);
    gemm(blas, w.v_proj, s.h, s.v, KD, H, B);
    launch_rope(s.q, d_positions, B, cfg.num_q_heads, cfg.head_dim, cfg.rope_theta, stream);
    launch_rope(s.k, d_positions, B, cfg.num_kv_heads, cfg.head_dim, cfg.rope_theta, stream);
    launch_write_kv_paged(s.k, s.v, k_pool, v_pool, d_block_table, d_lens, B, cfg.num_kv_heads, cfg.head_dim, block_size, max_blocks, stream);
    launch_paged_attention_decode(
        s.q, k_pool, v_pool, d_block_table, d_lens, s.attn,
        B, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
        block_size, max_blocks, max_len,
        1.0f / sqrtf((float)cfg.head_dim), stream);    
    gemm(blas, w.o_proj, s.attn, s.proj, H, QD, B);
    launch_residual_add(x, s.proj, B*H, stream);    
    // ── MLP block ────────────────────────────────────────────────────────────
    // TODO
    //  7. launch_rmsnorm(x, w.post_norm, s.h, B, H, cfg.rms_eps, stream)
    //  8. gemm(gate_proj → s.gate, I, H, B) ; gemm(up_proj → s.up, I, H, B)
    //  9. launch_silu_mul(s.gate, s.up, s.act, B, I, stream)
    // 10. gemm(blas, w.down_proj, s.act, s.proj, H, I, B)
    //     launch_residual_add(x, s.proj, B*H, stream)
    launch_rmsnorm(x, w.post_norm, s.h, B, H, cfg.rms_eps, stream);
    gemm(blas, w.gate_proj, s.h, s.gate, I, H, B);
    gemm(blas, w.up_proj, s.h, s.up, I, H, B);
    launch_silu_mul(s.gate, s.up, s.act, B, I, stream);
    gemm(blas, w.down_proj, s.act, s.proj, H, I, B);
    launch_residual_add(x, s.proj, B*H, stream);
}

// ── Full model decode step: ids → logits ─────────────────────────────────────
inline void forward_decode(const ModelConfig& cfg, const Weights& weights,
                           cublasHandle_t blas, DecodeScratch& s,
                           const int* d_token_ids,      // [B]
                           __half* x,                   // [B, hidden] scratch
                           __half* logits,              // [B, vocab]
                           __half* k_pool, __half* v_pool,
                           const int* d_block_table, const int* d_lens,
                           const int* d_positions,
                           int B, int block_size,
                           int num_blocks,    // TOTAL blocks in the pool
                           int max_blocks,    // block-table WIDTH (per sequence)
                           int max_len,
                           cudaStream_t stream) {
    // The pool is [num_layers, num_blocks, Hkv, BS, D], so the per-layer stride
    // uses num_blocks — NOT max_blocks, which is only how many blocks one
    // sequence may own. They are equal in the single-sequence verify harness
    // and very different in the server (thousands vs ~128), so mixing them up
    // silently reads the wrong pool region for every layer past 0.
    const size_t pool_per_layer = (size_t)num_blocks * cfg.num_kv_heads
                                * block_size * cfg.head_dim;

    launch_embedding(weights.embed_tokens, d_token_ids, x, B, cfg.hidden_dim, stream);
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const LayerWeights& w = weights.layers[layer];
        const size_t layer_offset = (size_t)layer * pool_per_layer;
        forward_layer(cfg, w, blas, s, x,
                      k_pool + layer_offset, v_pool + layer_offset,
                      d_block_table, d_lens, d_positions,
                      B, block_size, max_blocks, max_len, stream);
    }
    launch_rmsnorm(x, weights.final_norm, s.h, B, cfg.hidden_dim, cfg.rms_eps, stream);
    gemm(blas, weights.lm_head, s.h, logits, cfg.vocab_size, cfg.hidden_dim, B);            
}
