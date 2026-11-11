/**
 * weights.h — load the raw fp16 tensors written by tools/export_weights.py.
 *
 * No manifest parsing: file names are conventional and every shape is derivable
 * from ModelConfig, so the loader just checks that each file is the size it
 * should be. A mismatch means the export and the config disagree — worth
 * failing loudly on, because the alternative is silently reading a wrong-shaped
 * matrix and getting fluent nonsense.
 *
 * ── Layout convention ────────────────────────────────────────────────────────
 * Projections were TRANSPOSED at export to [in, out] row-major, which cuBLAS
 * reads directly as a column-major [out, in] matrix. So every GEMM here is
 * CUBLAS_OP_N / CUBLAS_OP_N:
 *
 *     y[out, B] = W[out, in] * x[in, B]
 *     cublasHgemm(N, N, m=out, n=B, k=in, W, lda=out, x, ldb=in, y, ldc=out)
 *
 * `lda = out` because the file holds [in, out] row-major = [out, in]
 * column-major with leading dimension `out`.
 */

#pragma once

#include "model_config.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

struct LayerWeights {
    __half* input_norm  = nullptr;   // [hidden]
    __half* q_proj      = nullptr;   // [hidden, n_q*head_dim]
    __half* k_proj      = nullptr;   // [hidden, n_kv*head_dim]
    __half* v_proj      = nullptr;   // [hidden, n_kv*head_dim]
    __half* o_proj      = nullptr;   // [n_q*head_dim, hidden]
    __half* post_norm   = nullptr;   // [hidden]
    __half* gate_proj   = nullptr;   // [hidden, inter]
    __half* up_proj     = nullptr;   // [hidden, inter]
    __half* down_proj   = nullptr;   // [inter, hidden]
};

class Weights {
public:
    void load(const std::string& dir, const ModelConfig& cfg) {
        cfg_ = cfg;
        dir_ = dir;
        const int H  = cfg.hidden_dim;
        const int QD = cfg.num_q_heads  * cfg.head_dim;
        const int KD = cfg.num_kv_heads * cfg.head_dim;
        const int I  = cfg.inter_dim;
        if (I <= 0) throw std::runtime_error("config.json has no intermediate_size");

        embed_tokens = load_one("embed_tokens", (size_t)cfg.vocab_size * H);
        final_norm   = load_one("model.norm",   (size_t)H);
        lm_head      = load_one("lm_head",      (size_t)H * cfg.vocab_size);

        layers.resize(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l) {
            const std::string t = tag(l);
            LayerWeights& w = layers[l];
            w.input_norm = load_one(t + ".input_layernorm", (size_t)H);
            w.q_proj     = load_one(t + ".q_proj",   (size_t)H * QD);
            w.k_proj     = load_one(t + ".k_proj",   (size_t)H * KD);
            w.v_proj     = load_one(t + ".v_proj",   (size_t)H * KD);
            w.o_proj     = load_one(t + ".o_proj",   (size_t)QD * H);
            w.post_norm  = load_one(t + ".post_attention_layernorm", (size_t)H);
            w.gate_proj  = load_one(t + ".gate_proj", (size_t)H * I);
            w.up_proj    = load_one(t + ".up_proj",   (size_t)H * I);
            w.down_proj  = load_one(t + ".down_proj", (size_t)I * H);
        }
        printf("Weights:   %d layers, %.2f GB on device\n",
               cfg.num_layers, bytes_ / 1e9);
    }

    void free() {
        for (auto p : owned_) cudaFree(p);
        owned_.clear();
        layers.clear();
    }

    __half* embed_tokens = nullptr;
    __half* final_norm   = nullptr;
    __half* lm_head      = nullptr;
    std::vector<LayerWeights> layers;

private:
    static std::string tag(int l) {
        char buf[16];
        snprintf(buf, sizeof buf, "layer%02d", l);
        return buf;
    }

    __half* load_one(const std::string& name, size_t n_elems) {
        const std::string path = dir_ + "/" + name + ".bin";
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);

        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        const size_t want = n_elems * sizeof(__half);
        if ((size_t)sz != want) {
            fclose(f);
            throw std::runtime_error(path + ": expected " + std::to_string(want)
                                     + " bytes, file has " + std::to_string(sz)
                                     + " — export and config disagree");
        }

        std::vector<__half> host(n_elems);
        if (fread(host.data(), sizeof(__half), n_elems, f) != n_elems) {
            fclose(f);
            throw std::runtime_error("short read: " + path);
        }
        fclose(f);

        __half* d = nullptr;
        if (cudaMalloc(&d, want) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed for " + name);
        if (cudaMemcpy(d, host.data(), want, cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("cudaMemcpy failed for " + name);

        owned_.push_back(d);
        bytes_ += want;
        return d;
    }

    ModelConfig cfg_{};
    std::string dir_;
    std::vector<__half*> owned_;
    double bytes_ = 0;
};
