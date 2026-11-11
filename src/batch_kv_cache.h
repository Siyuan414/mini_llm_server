/**
 * batch_kv_cache.h — N-slot KV cache for continuous batching (Stage 2A).
 *
 * ── Why this replaces Stage 1's ping-pong ────────────────────────────────────
 * Ping-pong worked because `past` grew by exactly 1 each step, which is the
 * stride TRT's `present` output writes. With N slots that breaks: `past` is
 * shared by the whole batch, so admitting a 500-token request when past=30
 * changes the row stride for EVERY slot and invalidates their layout.
 *
 * So we own the memory with a CONSTANT stride instead:
 *
 *     cache: [max_batch, H, max_seq, D]     stride = max_seq, never changes
 *
 * A constant stride makes slots independent — a new request is written into row
 * i without touching anyone else. The price is that we always pass
 * `past = max_seq`, so attention scans the full window every step regardless of
 * real lengths (~25% slower at 2048 ctx, per benchmarks/STAGE1.md). Eliminating
 * that waste is exactly what Stage 2B's paged cache is for.
 *
 * Per step: bind() → enqueue → commit_step() (scatter kernel + lens += 1).
 *
 * ── Batch dimension ──────────────────────────────────────────────────────────
 * TRT reads batch rows 0..B-1 contiguously, so B = highest_active_slot + 1.
 * A finished slot in the middle leaves a hole that still costs compute until a
 * new request refills it. Accepted for 2A: no copies, mild waste at steady
 * state. (Alternatives: run all max_batch rows, or compact on eviction at ~67MB
 * per eviction.)
 */

#pragma once

#include "kv_scatter.cuh"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

class BatchKVCache {
public:
    struct Config {
        int num_layers   = 0;
        int num_kv_heads = 0;
        int head_dim     = 0;
        int max_seq      = 0;   // per-slot capacity
        int max_batch    = 0;   // number of slots
    };

    // ── Lifecycle ────────────────────────────────────────────────────────────
    void alloc(const Config& cfg) {
        cfg_ = cfg;
        // TODO
        //  - k_[layer], v_[layer] : each cudaMalloc(max_batch*H*max_seq*D*2 bytes)
        //  - d_lens_              : int[max_batch] on device (the scatter kernel
        //                           reads this; it must live on the GPU)
        //  - present_k_[layer], present_v_[layer] : TRT writes here each step,
        //                           sized for past = max_seq, so max_seq+1 tokens
        //  - lens_ (host vector), active_ (host vector<bool>), all zeroed
        k_.resize(cfg_.num_layers);   
        v_.resize(cfg_.num_layers);
        present_k_.resize(cfg_.num_layers);  
        present_v_.resize(cfg_.num_layers);
        lens_.resize(cfg_.max_batch, 0);     
        active_.resize(cfg_.max_batch, 0);
        const size_t cache_elems   = (size_t)cfg_.max_batch * cfg_.num_kv_heads
                           * cfg_.max_seq * cfg_.head_dim;
        const size_t present_elems = (size_t)cfg_.max_batch * cfg_.num_kv_heads
                                * (cfg_.max_seq + 1) * cfg_.head_dim;

        
        for (int l = 0; l < cfg_.num_layers; ++l) {
            cudaMalloc(&k_[l],         cache_elems   * sizeof(__half));
            cudaMalloc(&v_[l],         cache_elems   * sizeof(__half));
            cudaMalloc(&present_k_[l], present_elems * sizeof(__half));
            cudaMalloc(&present_v_[l], present_elems * sizeof(__half));
        }
        cudaMalloc(&d_lens_, cfg_.max_batch * sizeof(int));
        cudaMemset(d_lens_, 0, cfg_.max_batch * sizeof(int));
    }

    void free() {
        // TODO: cudaFree everything
        for (auto& ptr : k_) {
            cudaFree(ptr);
        }
        for (auto& ptr : v_) {
            cudaFree(ptr);
        }
        for (auto& ptr : present_k_) {
            cudaFree(ptr);
        }
        for (auto& ptr : present_v_) {
            cudaFree(ptr);
        }
        cudaFree(d_lens_);
    }

    // ── Slot management (called by the scheduler) ────────────────────────────

    /// Claim a free slot. Returns the slot index, or -1 if none available.
    int acquire() {
        // TODO: find first !active_[i], mark active, lens_[i] = 0, return i
        for (int i = 0; i < cfg_.max_batch; ++i) {
            if (!active_[i]) {
                active_[i] = true;
                lens_[i] = 0;
                return i;
            }
        }
        return -1;
    }

    /// Release a finished slot. Its KV is NOT cleared — lens=0 plus the mask
    /// makes stale data unreachable, so zeroing would be wasted bandwidth.
    void release(int slot) {
        if(slot < 0 || slot >= cfg_.max_batch) {
            throw std::out_of_range("release() slot out of range");
        }
        active_[slot] = false;
        lens_[slot] = 0;
    }

    bool has_free() const { /* TODO */ 
        for(int i = 0; i < cfg_.max_batch; ++i) {
            if (!active_[i]) {
                return true;
            }
        }
        return false; }
    int  length(int slot) const { return lens_[slot]; }
    bool active(int slot) const { return active_[slot]; }

    /// TRT batch dimension: highest active slot + 1 (0 if none active).
    int batch_size() const {
        // TODO
        for (int i = cfg_.max_batch - 1; i >= 0; --i) {
            if (active_[i]) {
                return i + 1;
            }
        }
        return 0;
    }

    // ── Admission: install a prefilled request's KV into a slot ─────────────
    /// After prefilling a new request separately (batch 1, profile 0, past=0),
    /// its KV comes back as [1, H, n_tokens, D] per layer. Copy it into row
    /// `slot` of our cache, where the stride is max_seq rather than n_tokens.
    ///
    /// Per (layer, head) this is a contiguous run of n_tokens*D halves, so a
    /// cudaMemcpy2DAsync with spitch = n_tokens*D*2 and dpitch = max_seq*D*2
    /// handles all H heads in one call.
    void install_prefill(int slot, int n_tokens,
                         const std::vector<__half*>& prefill_k,
                         const std::vector<__half*>& prefill_v,
                         cudaStream_t stream) {
        // TODO
        //  for each layer: cudaMemcpy2DAsync(present_k_[l], k_[l], n_tokens*D*2, max_seq*D*2, stream)
        //                  same for v  
        for(int l = 0; l < cfg_.num_layers; ++l) {
            size_t width_bytes = n_tokens * cfg_.head_dim * sizeof(__half);
            size_t height = cfg_.num_kv_heads;
            size_t src_pitch = width_bytes; // contiguous for H heads
            size_t dst_pitch = cfg_.max_seq * cfg_.head_dim * sizeof(__half);
            cudaMemcpy2DAsync(k_[l] + slot *  height * dst_pitch / sizeof(__half), dst_pitch,
                              prefill_k[l], src_pitch,
                              width_bytes, height,
                              cudaMemcpyDeviceToDevice, stream);
            cudaMemcpy2DAsync(v_[l] + slot * height * dst_pitch / sizeof(__half), dst_pitch,
                              prefill_v[l], src_pitch,
                              width_bytes, height,
                              cudaMemcpyDeviceToDevice, stream);
        }
        // Without this the slot has KV but the cache thinks it is empty, and the
        // first scatter would overwrite token 0. sync_lens pushes it to device.
        lens_[slot] = n_tokens;
        sync_lens(stream);
    }

    // ── Per-step ─────────────────────────────────────────────────────────────

    /// Upload host lens_ to d_lens_ (the scatter kernel needs them on device).
    void sync_lens(cudaStream_t stream) {
        // TODO
        cudaMemcpyAsync(d_lens_, lens_.data(), cfg_.max_batch * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    /// Bind our cache as `past` and our scratch as `present`, for all layers.
    /// past shape is [B, H, max_seq, D] — always max_seq, never the real length.
    void bind(nvinfer1::IExecutionContext* ctx,
              const std::vector<std::string>& past_k,
              const std::vector<std::string>& past_v,
              const std::vector<std::string>& present_k,
              const std::vector<std::string>& present_v) const {
        // TODO
        //  for each layer: ctx->setBindingDimensions(past_k[l], nvinfer1::Dims4{cfg_.max_batch, cfg_.num_kv_heads, cfg_.max_seq, cfg_.head_dim});
        //                  same for past_v, present_k, present_v   
        const int B = batch_size();
        nvinfer1::Dims4 past_shape{B, cfg_.num_kv_heads, cfg_.max_seq, cfg_.head_dim};
        for (int l = 0; l < cfg_.num_layers; ++l) {
            ctx->setInputShape(past_k[l].c_str(), past_shape);
            ctx->setInputShape(past_v[l].c_str(), past_shape);
            ctx->setTensorAddress(past_k[l].c_str(), k_[l]);
            ctx->setTensorAddress(past_v[l].c_str(), v_[l]);
            ctx->setTensorAddress(present_k[l].c_str(), present_k_[l]);
            ctx->setTensorAddress(present_v[l].c_str(), present_v_[l]);
        }
    }

    /// After a successful enqueue: scatter each slot's new token from `present`
    /// into cache[slot, :, lens[slot], :], then advance lens.
    void commit_step(cudaStream_t stream) {
        // TODO
        //  for each layer: launch_scatter_new_kv(present_k_[l], k_[l], d_lens_,
        //                      B, H, D, /*past=*/max_seq, max_seq, stream)
        //                  same for v
        //  then lens_[i] += 1 for active slots, and sync_lens()
        //  ORDER MATTERS: scatter must use the OLD lens (the write position),
        //  so increment only after launching.
        // B must match what bind() declared — max_batch would scatter rows TRT
        // never wrote this step.
        const int B = batch_size();
        if (B == 0) return;

        for(int l = 0; l < cfg_.num_layers; ++l) {
            launch_scatter_new_kv(present_k_[l], k_[l], d_lens_,
                                  B, cfg_.num_kv_heads, cfg_.head_dim,
                                  /*past=*/cfg_.max_seq, cfg_.max_seq, stream);
            launch_scatter_new_kv(present_v_[l], v_[l], d_lens_,
                                  B, cfg_.num_kv_heads, cfg_.head_dim,
                                  /*past=*/cfg_.max_seq, cfg_.max_seq, stream);
        }
        for(int i = 0; i < B; ++i) {
            if(active_[i] && lens_[i] < cfg_.max_seq) {
                lens_[i] += 1;
            }
        }
        // Without this d_lens_ still holds last step's values and the next
        // scatter writes to the wrong position — quiet, like the S1 stride bug.
        sync_lens(stream);
    }

    // ── Batched input construction (host side, uploaded each step) ───────────

    /// mask[b] = [1 x lens[b]] [0 x (max_seq - lens[b])] [1 x seq]
    /// Row b's real tokens, then dead padding, then the new token(s) — TRT
    /// appends new KV at index `past`, so the new columns are always last.
    std::vector<int64_t> build_mask(int seq) const {
        // TODO: returns [B * (max_seq + seq)]
        // build a vector of size cfg_.max_batch * (cfg_.max_seq + seq), 
        // fill it with 0s, then for each active slot b, 
        // set the first lens_[b] elements to 1, and the last seq elements to 1.
        const int B = batch_size();   // must match bind(); max_batch would make
                                      // the uploaded tensor the wrong size
        std::vector<int64_t> mask((size_t)B * (cfg_.max_seq + seq), 0);
        for(int b = 0; b < B; ++b) {
            if(active_[b]) {
                // Set the first lens_[b] elements to 1
                for(int i = 0; i < lens_[b]; ++i) {
                    mask[b * (cfg_.max_seq + seq) + i] = 1;
                }
                // Set the last seq elements to 1
                for(int i = 0; i < seq; ++i) {
                    mask[b * (cfg_.max_seq + seq) + (cfg_.max_seq + i)] = 1;
                }
            }
        }
        return mask;
    }

    /// pos[b] = [lens[b], lens[b]+1, ...] — each slot's OWN next position.
    /// Passing `past` here instead would misrotate RoPE.
    std::vector<int64_t> build_positions(int seq) const {
        // TODO: returns [B * seq]
        const int B = batch_size();
        std::vector<int64_t> positions((size_t)B * seq, 0);
        for(int b = 0; b < B; ++b) {
            if(active_[b]) {
                for(int i = 0; i < seq; ++i) {
                    positions[b * seq + i] = lens_[b] + i;
                }
            }
        }
        return positions;
    }

    size_t bytes() const {
        return (size_t)cfg_.max_batch * cfg_.num_kv_heads * cfg_.max_seq
             * cfg_.head_dim * sizeof(__half) * 2 * cfg_.num_layers;
    }

private:
    Config cfg_{};
    std::vector<__half*> k_, v_;                 // [layers], each [B,H,max_seq,D]
    std::vector<__half*> present_k_, present_v_; // [layers], TRT scratch
    int* d_lens_ = nullptr;                      // [max_batch] on device
    std::vector<int>  lens_;                     // host mirror
    std::vector<char> active_;                   // vector<bool> has no data()
};
