/**
 * kv_cache.h — KV cache manager (header-only)
 *
 * Owns all KV device memory and knows how to bind it to a TRT execution context.
 * The runtime never touches a raw KV pointer.
 *
 * ── Layout, and why it is what it is ─────────────────────────────────────────
 * TRT reads `past_key_values.i.key` as a CONTIGUOUS [1, H, past_len, D] tensor:
 * head h begins at h*past_len*D. A fixed-capacity buffer strided by max_seq
 * therefore CANNOT be handed to TRT directly — every head but head 0 would be
 * misread, silently, producing plausible-looking garbage.
 *
 * The `present.i.key` output, however, is already exactly that contiguous layout
 * for (past_len + n) tokens. So instead of copying new entries into a persistent
 * buffer, we PING-PONG: the buffer TRT just wrote as `present` becomes the next
 * step's `past`. Zero copies per step, and always the layout TRT expects.
 *
 * ── Stage 2 seam ─────────────────────────────────────────────────────────────
 * The runtime only calls: fits() / bind() / commit() / length() / reset().
 * A paged implementation (fixed-size blocks + block table + allocator) replaces
 * the body of this class without the engine loop changing. bind() is where a
 * paged version would instead publish a block table to a custom attention op.
 */

#pragma once

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

class KVCache {
public:
    struct Config {
        int num_layers   = 0;
        int num_kv_heads = 0;
        int head_dim     = 0;
        int max_seq      = 0;   // context window (KV capacity)
    };

    // ── Lifecycle ────────────────────────────────────────────────────────────
    void alloc(const Config& cfg) {
        cfg_ = cfg;
        const size_t buf_sz = per_layer_bytes();
        for (int b = 0; b < 2; ++b) {
            k_[b].resize(cfg_.num_layers);
            v_[b].resize(cfg_.num_layers);
            for (int i = 0; i < cfg_.num_layers; ++i) {
                ck(cudaMalloc(&k_[b][i], buf_sz));
                ck(cudaMalloc(&v_[b][i], buf_sz));
            }
        }
        // TRT rejects a null address even for a zero-length tensor, so past_len==0
        // still needs a valid pointer to bind.
        ck(cudaMalloc(&dummy_, sizeof(__half)));
        reset();
    }

    void free() {
        for (int b = 0; b < 2; ++b) {
            for (auto p : k_[b]) cudaFree(p);
            for (auto p : v_[b]) cudaFree(p);
            k_[b].clear();
            v_[b].clear();
        }
        cudaFree(dummy_);
        dummy_ = nullptr;
    }

    // ── State ────────────────────────────────────────────────────────────────
    void reset()             { len_ = 0; cur_ = 0; }
    int  length()      const { return len_; }
    int  capacity()    const { return cfg_.max_seq; }
    bool fits(int n)   const { return len_ + n <= cfg_.max_seq; }

    size_t bytes() const {
        return per_layer_bytes() * cfg_.num_layers * 2 /*K,V*/ * 2 /*ping-pong*/;
    }

    // ── Binding ──────────────────────────────────────────────────────────────
    // Point the context's past inputs at the current buffer and its present
    // outputs at the other one. Call before enqueue; call commit(n) after.
    void bind(nvinfer1::IExecutionContext* ctx,
              const std::vector<std::string>& past_k_names,
              const std::vector<std::string>& past_v_names,
              const std::vector<std::string>& present_k_names,
              const std::vector<std::string>& present_v_names) const {
        const int src = cur_, dst = cur_ ^ 1;
        const nvinfer1::Dims4 past_shape{1, cfg_.num_kv_heads, len_, cfg_.head_dim};

        for (int i = 0; i < cfg_.num_layers; ++i) {
            ctx->setInputShape(past_k_names[i].c_str(), past_shape);
            ctx->setInputShape(past_v_names[i].c_str(), past_shape);
            ctx->setTensorAddress(past_k_names[i].c_str(),
                                  len_ == 0 ? (void*)dummy_ : (void*)k_[src][i]);
            ctx->setTensorAddress(past_v_names[i].c_str(),
                                  len_ == 0 ? (void*)dummy_ : (void*)v_[src][i]);
            ctx->setTensorAddress(present_k_names[i].c_str(), k_[dst][i]);
            ctx->setTensorAddress(present_v_names[i].c_str(), v_[dst][i]);
        }
    }

    // Advance after a successful enqueue of n tokens: present becomes past.
    void commit(int n) {
        if (!fits(n)) throw std::runtime_error("KVCache: capacity exceeded");
        cur_ ^= 1;
        len_ += n;
    }

private:
    static void ck(cudaError_t e) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("KVCache CUDA: ")
                                     + cudaGetErrorString(e));
    }

    size_t per_layer_bytes() const {
        return (size_t)cfg_.num_kv_heads * cfg_.max_seq * cfg_.head_dim
               * sizeof(__half);
    }

    Config cfg_{};
    std::vector<__half*> k_[2], v_[2];
    __half* dummy_ = nullptr;
    int cur_ = 0;   // which ping-pong buffer holds the current past
    int len_ = 0;   // tokens currently cached
};
