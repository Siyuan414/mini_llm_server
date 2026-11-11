/**
 * varlen_layer.h — forward pass over a MIXED batch of prefill chunks and
 * decode tokens (S3.4).
 *
 * decode_layer.h assumes one query token per sequence. This is the same layer
 * with that assumption removed: the batch is `total_q` query tokens drawn from
 * any number of sequences in any proportion, described by
 *
 *     q_seq[g]   which sequence query token g belongs to
 *     q_pos[g]   its ABSOLUTE position in that sequence
 *
 * Decode is total_q == n_seqs with q_pos[g] == len-1, so this subsumes
 * forward_decode rather than sitting beside it.
 *
 * ── Three things change, beyond the attention kernel ─────────────────────────
 * 1. RoPE takes ABSOLUTE positions. Query 0 of a second chunk is at position
 *    236, not 0. launch_rope already takes a per-row position array, so passing
 *    q_pos works unchanged — but passing chunk-relative indices would produce
 *    fluent-and-wrong text, the hardest failure in this system to spot.
 *
 * 2. write_kv stores n tokens per sequence, at their own positions. The decode
 *    version derived one position from lens[b]-1; here every query token has
 *    its own, and all of them must land BEFORE attention runs, because query p
 *    attends to key p.
 *
 * 3. lm_head runs on the LAST query token of each sequence only. A prefill
 *    chunk's interior tokens predict nothing we keep — their job is to populate
 *    the KV cache. Running lm_head over all 512 rows would cost a
 *    512 x 128256 GEMM (262 MB of fp16 logits) to throw away 511/512 of it, so
 *    the final hidden states are gathered to [n_out, hidden] first.
 *
 *    Careful: a sequence still MID-prefill has no last token worth sampling
 *    either — it will get more prompt next step. Only sequences whose chunk
 *    reaches the end of their prompt belong in the gather list. That decision
 *    is the caller's; this file just honours the index array.
 */

#pragma once

#include "decode_layer.h"                 // gemm, DecodeScratch, layer kernels
#include "paged_attention_varlen.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ── Write a whole batch of K/V into the paged pool ───────────────────────────
// One thread block per (query token, kv_head). Mirrors the attention kernel's
// indirection in the opposite direction.
__global__ void write_kv_paged_varlen(
        const __half* __restrict__ k,            // [total_q, Hkv, D]
        const __half* __restrict__ v,
        __half* __restrict__ k_pool,
        __half* __restrict__ v_pool,
        const int* __restrict__ block_table,     // [n_seqs, max_blocks]
        const int* __restrict__ q_seq,           // [total_q]
        const int* __restrict__ q_pos,           // [total_q]
        int Hkv, int D, int BS, int max_blocks) {

    const int g   = blockIdx.x / Hkv;
    const int kvh = blockIdx.x % Hkv;
    const int b   = q_seq[g];
    const int pos = q_pos[g];                    // absolute, not lens[b]-1

    const int logical  = pos / BS;
    const int offset   = pos % BS;
    const int physical = block_table[(size_t)b * max_blocks + logical];
    if (physical < 0) return;                    // allocator/kernel disagree

    const size_t dst = ((size_t)physical * Hkv + kvh) * BS * D + (size_t)offset * D;
    const size_t src = ((size_t)g * Hkv + kvh) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        k_pool[dst + i] = k[src + i];
        v_pool[dst + i] = v[src + i];
    }
}

inline void launch_write_kv_paged_varlen(
        const __half* k, const __half* v,
        __half* k_pool, __half* v_pool,
        const int* block_table, const int* q_seq, const int* q_pos,
        int total_q, int Hkv, int D, int BS, int max_blocks, cudaStream_t s) {
    write_kv_paged_varlen<<<total_q * Hkv, 256, 0, s>>>(
        k, v, k_pool, v_pool, block_table, q_seq, q_pos,
        Hkv, D, BS, max_blocks);
}

// ── Gather selected rows: out[i, :] = in[idx[i], :] ──────────────────────────
__global__ void gather_rows(const __half* __restrict__ in,
                            const int* __restrict__ idx,
                            __half* __restrict__ out,
                            int n, int width) {
    const int i = blockIdx.x;
    if (i >= n) return;
    const size_t src = (size_t)idx[i] * width;
    const size_t dst = (size_t)i * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        out[dst + j] = in[src + j];
}

inline void launch_gather_rows(const __half* in, const int* idx, __half* out,
                               int n, int width, cudaStream_t s) {
    if (n > 0) gather_rows<<<n, 256, 0, s>>>(in, idx, out, n, width);
}

// ── One decoder layer over a mixed batch ────────────────────────────────────
inline void forward_layer_varlen(
        const ModelConfig& cfg, const LayerWeights& w,
        cublasHandle_t blas, DecodeScratch& s,
        __half* x,                              // [total_q, hidden] in/out
        __half* k_pool, __half* v_pool,
        const int* d_block_table,
        const int* d_q_seq, const int* d_q_pos,
        int total_q, int block_size, int max_blocks, int max_scan,
        cudaStream_t stream) {

    const int H  = cfg.hidden_dim;
    const int QD = cfg.num_q_heads  * cfg.head_dim;
    const int KD = cfg.num_kv_heads * cfg.head_dim;
    const int I  = cfg.inter_dim;

    // ── Attention block ─────────────────────────────────────────────────────
    launch_rmsnorm(x, w.input_norm, s.h, total_q, H, cfg.rms_eps, stream);
    gemm(blas, w.q_proj, s.h, s.q, QD, H, total_q);
    gemm(blas, w.k_proj, s.h, s.k, KD, H, total_q);
    gemm(blas, w.v_proj, s.h, s.v, KD, H, total_q);

    // ABSOLUTE positions — q_pos, never a chunk-relative index.
    launch_rope(s.q, d_q_pos, total_q, cfg.num_q_heads,  cfg.head_dim,
                cfg.rope_theta, stream);
    launch_rope(s.k, d_q_pos, total_q, cfg.num_kv_heads, cfg.head_dim,
                cfg.rope_theta, stream);

    // BEFORE attention: query p attends to key p, so every token in this batch
    // must already be in the pool.
    launch_write_kv_paged_varlen(s.k, s.v, k_pool, v_pool, d_block_table,
                                 d_q_seq, d_q_pos, total_q,
                                 cfg.num_kv_heads, cfg.head_dim,
                                 block_size, max_blocks, stream);

    launch_paged_attention_varlen(
        s.q, k_pool, v_pool, d_block_table, d_q_seq, d_q_pos, s.attn,
        total_q, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
        block_size, max_blocks, max_scan,
        1.0f / sqrtf((float)cfg.head_dim), stream);

    gemm(blas, w.o_proj, s.attn, s.proj, H, QD, total_q);
    launch_residual_add(x, s.proj, total_q * H, stream);

    // ── MLP block ───────────────────────────────────────────────────────────
    launch_rmsnorm(x, w.post_norm, s.h, total_q, H, cfg.rms_eps, stream);
    gemm(blas, w.gate_proj, s.h, s.gate, I, H, total_q);
    gemm(blas, w.up_proj,   s.h, s.up,   I, H, total_q);
    launch_silu_mul(s.gate, s.up, s.act, total_q, I, stream);
    gemm(blas, w.down_proj, s.act, s.proj, H, I, total_q);
    launch_residual_add(x, s.proj, total_q * H, stream);
}

// ── Full forward pass: mixed batch of tokens → logits for selected rows ──────
/// `d_out_rows` holds `n_out` GLOBAL query indices (into [0, total_q)) — the
/// last query token of each sequence that should be sampled this step. Pass
/// n_out == total_q with d_out_rows == {0,1,...} for a pure decode step.
///
/// `logits` is [n_out, vocab], NOT [total_q, vocab].
inline void forward_varlen(const ModelConfig& cfg, const Weights& weights,
                           cublasHandle_t blas, DecodeScratch& s,
                           const int* d_token_ids,     // [total_q]
                           __half* x,                  // [total_q, hidden]
                           __half* logits,             // [n_out, vocab]
                           __half* k_pool, __half* v_pool,
                           const int* d_block_table,
                           const int* d_q_seq, const int* d_q_pos,
                           const int* d_out_rows,      // [n_out]
                           int total_q, int n_out, int block_size,
                           int num_blocks,   // TOTAL blocks in the pool
                           int max_blocks,   // block-table WIDTH per sequence
                           int max_scan,     // 1 + max(q_pos)
                           cudaStream_t stream) {
    // Pool stride uses num_blocks (pool size), NOT max_blocks (table width).
    // They coincide in single-sequence harnesses and differ by orders of
    // magnitude in the server; mixing them reads the wrong region for every
    // layer past 0.
    const size_t pool_per_layer = (size_t)num_blocks * cfg.num_kv_heads
                                * block_size * cfg.head_dim;

    launch_embedding(weights.embed_tokens, d_token_ids, x,
                     total_q, cfg.hidden_dim, stream);

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const size_t off = (size_t)layer * pool_per_layer;
        forward_layer_varlen(cfg, weights.layers[layer], blas, s, x,
                             k_pool + off, v_pool + off,
                             d_block_table, d_q_seq, d_q_pos,
                             total_q, block_size, max_blocks, max_scan, stream);
    }

    // Normalize every row, then keep only the rows that predict something.
    // Gathering AFTER the norm (rather than before) keeps the norm a single
    // launch over contiguous memory and costs one extra copy of n_out rows.
    launch_rmsnorm(x, weights.final_norm, s.h, total_q, cfg.hidden_dim,
                   cfg.rms_eps, stream);
    launch_gather_rows(s.h, d_out_rows, s.h_last, n_out, cfg.hidden_dim, stream);
    gemm(blas, weights.lm_head, s.h_last, logits,
         cfg.vocab_size, cfg.hidden_dim, n_out);
}
