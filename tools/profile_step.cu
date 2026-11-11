/**
 * profile_step.cu — what fraction of a forward step is attention? (S4.0)
 *
 * bench_varlen_attention.cu measures the kernel in isolation. This measures its
 * SHARE of a real step, which is the number that decides whether optimizing it
 * is worth anything: if attention is 20% of the step, a 2x kernel speedup buys
 * 10% end to end and S4.2's week of tiling work is not justified.
 *
 * Amdahl, applied honestly before writing the optimization rather than after.
 *
 * The layer body is duplicated from varlen_layer.h with CUDA events around each
 * category. Duplication is deliberate — production code stays free of
 * instrumentation, and the copy is short enough to eyeball against the original.
 *
 * Four categories:
 *   attention    the paged varlen kernel
 *   gemm         7 cuBLAS calls per layer (qkv, o, gate, up, down)
 *   elementwise  rmsnorm x2, rope x2, write_kv, silu_mul, residual x2
 *   lm_head      final norm + gather + vocab projection (once per step)
 *
 * Build:
 *   nvcc -std=c++17 -O3 -I src -I kernels tools/profile_step.cu \
 *        -o build/profile_step -lcublas --extended-lambda -arch=sm_120
 * Run:
 *   ./build/profile_step --weights weights --config onnx/config.json \
 *                        --n-seqs 32 --q-len 1 --context 512
 *   ./build/profile_step ... --n-seqs 1 --q-len 256 --context 0
 */

#include "model_config.h"
#include "weights.h"
#include "varlen_layer.h"

#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                   \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    exit(1);} } while(0)

// ── A stopwatch that accumulates elapsed time per category ──────────────────
struct Timer {
    cudaEvent_t a, b;
    double total_ms = 0;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    void stop()  {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        total_ms += ms;
    }
    void reset() { total_ms = 0; }
};

int main(int argc, char** argv) {
    std::string wdir = "weights", cfg_path = "onnx/config.json";
    int n_seqs = 32, q_len = 1, context = 512, block_size = 16, iters = 50;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--weights" && i+1<argc) wdir     = argv[++i];
        else if (s == "--config"  && i+1<argc) cfg_path = argv[++i];
        else if (s == "--n-seqs"  && i+1<argc) n_seqs   = std::stoi(argv[++i]);
        else if (s == "--q-len"   && i+1<argc) q_len    = std::stoi(argv[++i]);
        else if (s == "--context" && i+1<argc) context  = std::stoi(argv[++i]);
        else if (s == "--iters"   && i+1<argc) iters    = std::stoi(argv[++i]);
    }

    ModelConfig cfg;  cfg.load(cfg_path, 4096);
    Weights weights;  weights.load(wdir, cfg);

    const int total_q    = n_seqs * q_len;
    const int max_len    = context + q_len;
    const int max_blocks = (max_len + block_size - 1) / block_size;
    const int num_blocks = n_seqs * max_blocks;

    std::vector<int> h_seq(total_q), h_pos(total_q), h_ids(total_q), h_out(n_seqs);
    int g = 0, max_scan = 0;
    for (int s = 0; s < n_seqs; ++s) {
        for (int i = 0; i < q_len; ++i, ++g) {
            h_seq[g] = s;
            h_pos[g] = context + i;
            h_ids[g] = 1000 + (g % 5000);
            max_scan = std::max(max_scan, h_pos[g] + 1);
        }
        h_out[s] = (s + 1) * q_len - 1;          // last query token of seq s
    }
    std::vector<int> h_tab((size_t)n_seqs * max_blocks);
    for (int s = 0; s < n_seqs; ++s)
        for (int b = 0; b < max_blocks; ++b)
            h_tab[(size_t)s * max_blocks + b] = s * max_blocks + b;

    const size_t pool_per_layer =
        (size_t)num_blocks * cfg.num_kv_heads * block_size * cfg.head_dim;

    __half *k_pool, *v_pool, *x, *logits;
    CK(cudaMalloc(&k_pool, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMalloc(&v_pool, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMalloc(&x,      (size_t)total_q * cfg.hidden_dim * sizeof(__half)));
    CK(cudaMalloc(&logits, (size_t)n_seqs  * cfg.vocab_size * sizeof(__half)));
    CK(cudaMemset(k_pool, 0x3C, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMemset(v_pool, 0x3C, pool_per_layer * cfg.num_layers * sizeof(__half)));

    int *d_seq, *d_pos, *d_ids, *d_tab, *d_out;
    CK(cudaMalloc(&d_seq, total_q*sizeof(int)));
    CK(cudaMalloc(&d_pos, total_q*sizeof(int)));
    CK(cudaMalloc(&d_ids, total_q*sizeof(int)));
    CK(cudaMalloc(&d_out, n_seqs *sizeof(int)));
    CK(cudaMalloc(&d_tab, h_tab.size()*sizeof(int)));
    CK(cudaMemcpy(d_seq, h_seq.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, h_pos.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ids, h_ids.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_out, h_out.data(), n_seqs *sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tab, h_tab.data(), h_tab.size()*sizeof(int),
                  cudaMemcpyHostToDevice));

    DecodeScratch s;  s.alloc(cfg, total_q, n_seqs);
    cublasHandle_t blas; cublasCreate(&blas);

    const int H  = cfg.hidden_dim;
    const int QD = cfg.num_q_heads  * cfg.head_dim;
    const int KD = cfg.num_kv_heads * cfg.head_dim;
    const int I  = cfg.inter_dim;
    const float scale = 1.0f / sqrtf((float)cfg.head_dim);

    Timer t_attn, t_gemm, t_elem, t_head;

    // One instrumented step, matching forward_varlen's structure.
    auto step = [&] {
        t_elem.start();
        launch_embedding(weights.embed_tokens, d_ids, x, total_q, H, 0);
        t_elem.stop();

        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            const LayerWeights& w = weights.layers[layer];
            const size_t off = (size_t)layer * pool_per_layer;
            __half* kp = k_pool + off;
            __half* vp = v_pool + off;

            t_elem.start();
            launch_rmsnorm(x, w.input_norm, s.h, total_q, H, cfg.rms_eps, 0);
            t_elem.stop();

            t_gemm.start();
            gemm(blas, w.q_proj, s.h, s.q, QD, H, total_q);
            gemm(blas, w.k_proj, s.h, s.k, KD, H, total_q);
            gemm(blas, w.v_proj, s.h, s.v, KD, H, total_q);
            t_gemm.stop();

            t_elem.start();
            launch_rope(s.q, d_pos, total_q, cfg.num_q_heads,  cfg.head_dim,
                        cfg.rope_theta, 0);
            launch_rope(s.k, d_pos, total_q, cfg.num_kv_heads, cfg.head_dim,
                        cfg.rope_theta, 0);
            launch_write_kv_paged_varlen(s.k, s.v, kp, vp, d_tab, d_seq, d_pos,
                                         total_q, cfg.num_kv_heads, cfg.head_dim,
                                         block_size, max_blocks, 0);
            t_elem.stop();

            t_attn.start();
            launch_paged_attention_varlen(s.q, kp, vp, d_tab, d_seq, d_pos, s.attn,
                                          total_q, cfg.num_q_heads, cfg.num_kv_heads,
                                          cfg.head_dim, block_size, max_blocks,
                                          max_scan, scale, 0);
            t_attn.stop();

            t_gemm.start();
            gemm(blas, w.o_proj, s.attn, s.proj, H, QD, total_q);
            t_gemm.stop();

            t_elem.start();
            launch_residual_add(x, s.proj, total_q * H, 0);
            launch_rmsnorm(x, w.post_norm, s.h, total_q, H, cfg.rms_eps, 0);
            t_elem.stop();

            t_gemm.start();
            gemm(blas, w.gate_proj, s.h, s.gate, I, H, total_q);
            gemm(blas, w.up_proj,   s.h, s.up,   I, H, total_q);
            t_gemm.stop();

            t_elem.start();
            launch_silu_mul(s.gate, s.up, s.act, total_q, I, 0);
            t_elem.stop();

            t_gemm.start();
            gemm(blas, w.down_proj, s.act, s.proj, H, I, total_q);
            t_gemm.stop();

            t_elem.start();
            launch_residual_add(x, s.proj, total_q * H, 0);
            t_elem.stop();
        }

        t_head.start();
        launch_rmsnorm(x, weights.final_norm, s.h, total_q, H, cfg.rms_eps, 0);
        launch_gather_rows(s.h, d_out, s.h_last, n_seqs, H, 0);
        gemm(blas, weights.lm_head, s.h_last, logits, cfg.vocab_size, H, n_seqs);
        t_head.stop();
    };

    for (int i = 0; i < 5; ++i) step();          // warm up
    CK(cudaDeviceSynchronize());
    t_attn.reset(); t_gemm.reset(); t_elem.reset(); t_head.reset();

    // Wall time of the whole loop, to expose event overhead.
    cudaEvent_t w0, w1;
    CK(cudaEventCreate(&w0)); CK(cudaEventCreate(&w1));
    CK(cudaEventRecord(w0));
    for (int i = 0; i < iters; ++i) step();
    CK(cudaEventRecord(w1));
    CK(cudaEventSynchronize(w1));
    float wall_ms = 0; CK(cudaEventElapsedTime(&wall_ms, w0, w1));

    const double n = iters;
    const double attn = t_attn.total_ms / n;
    const double gm   = t_gemm.total_ms / n;
    const double el   = t_elem.total_ms / n;
    const double hd   = t_head.total_ms / n;
    const double sum  = attn + gm + el + hd;
    const double wall = wall_ms / n;

    printf("\nshape: %d seqs x %d query tokens = %d total_q, context %d\n",
           n_seqs, q_len, total_q, context);
    printf("%d layers, %d iterations\n\n", cfg.num_layers, iters);
    printf("  %-14s %9s  %6s\n", "category", "ms/step", "share");
    printf("  %-14s %9.3f  %5.1f%%\n", "attention",   attn, 100*attn/sum);
    printf("  %-14s %9.3f  %5.1f%%\n", "gemm",        gm,   100*gm/sum);
    printf("  %-14s %9.3f  %5.1f%%\n", "elementwise", el,   100*el/sum);
    printf("  %-14s %9.3f  %5.1f%%\n", "lm_head",     hd,   100*hd/sum);
    printf("  %-14s %9.3f\n", "sum",  sum);
    printf("  %-14s %9.3f   (%.1f%% event overhead)\n", "wall", wall,
           100.0 * (wall - sum) / wall);

    // Amdahl: the most an attention-only optimization can ever buy.
    printf("\n  ceiling for an attention-only speedup:\n");
    for (double sp : {1.5, 2.0, 4.0}) {
        const double newsum = sum - attn + attn / sp;
        printf("    %.1fx faster attention -> %.2f ms/step, %.1f%% overall\n",
               sp, newsum, 100.0 * (sum / newsum - 1.0));
    }

    s.free(); weights.free(); cublasDestroy(blas);
    cudaFree(k_pool); cudaFree(v_pool); cudaFree(x); cudaFree(logits);
    cudaFree(d_seq); cudaFree(d_pos); cudaFree(d_ids); cudaFree(d_tab); cudaFree(d_out);
    return 0;
}
