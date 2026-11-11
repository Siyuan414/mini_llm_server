/**
 * verify_decode_path.cu — S2.7: compare the custom decode path against HF.
 *
 * Feeds the reference prompt through forward_layer ONE TOKEN AT A TIME (which
 * is what a decode step does) and, after the final token, compares the hidden
 * state after every layer against reference/layerNN.bin.
 *
 * Layer by layer, not end to end: if all 16 layers are wired and only the
 * logits are checked, a mismatch tells you nothing about where it broke.
 *
 *   python tools/dump_reference.py --model <hf_dir>
 *   nvcc -std=c++17 -I src -I kernels tools/verify_decode_path.cu \
 *        -o build/verify_decode -lcublas --extended-lambda -arch=sm_120
 *   ./build/verify_decode
 */

#include "model_config.h"
#include "weights.h"
#include "block_allocator.h"
#include "decode_layer.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                    \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));       \
    exit(1);} } while(0)

static std::vector<float> read_f32(const std::string& p, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { printf("cannot open %s\n", p.c_str()); exit(1); }
    std::vector<float> v(n);
    f.read((char*)v.data(), n * sizeof(float));
    if (!f) { printf("short read %s\n", p.c_str()); exit(1); }
    return v;
}

static std::vector<float> download(const __half* d, size_t n) {
    std::vector<__half> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = __half2float(h[i]);
    return f;
}

// Relative error is the right metric here: hidden-state magnitudes grow through
// the layers, so a fixed absolute tolerance would be too loose early and too
// tight late.
static double rel_err(const std::vector<float>& got,
                      const std::vector<float>& want, size_t off, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = got[i] - want[off + i];
        num += d * d;
        den += (double)want[off + i] * want[off + i];
    }
    return std::sqrt(num / (den + 1e-12));
}

int main(int argc, char** argv) {
    const std::string wdir = (argc > 1) ? argv[1] : "weights";
    const std::string rdir = (argc > 2) ? argv[2] : "reference";
    const std::string cdir = (argc > 3) ? argv[3] : "onnx";   // holds config.json

    ModelConfig cfg;
    cfg.load(cdir + "/config.json", 2048);
    cfg.print();

    Weights weights;
    weights.load(wdir, cfg);

    // ── tokens ───────────────────────────────────────────────────────────────
    std::vector<int> tokens;
    { std::ifstream f(rdir + "/tokens.txt");
      for (int t; f >> t; ) tokens.push_back(t); }
    const int N = (int)tokens.size();
    printf("Prompt:    %d tokens\n\n", N);

    // ── paged cache for ONE sequence ─────────────────────────────────────────
    const int BS = 16;
    const int max_blocks = (N + BS - 1) / BS + 2;
    const int num_blocks = max_blocks;
    BlockAllocator alloc;
    alloc.configure({BS, num_blocks, 0});
    alloc.allocate(1, N);                       // enough for the whole prompt
    const std::vector<int> table = alloc.flatten({1}, max_blocks);

    const size_t pool_per_layer = (size_t)num_blocks * cfg.num_kv_heads * BS
                                * cfg.head_dim;
    const size_t pool_elems     = pool_per_layer * cfg.num_layers;

    __half *k_pool, *v_pool, *x, *logits;
    int *d_table, *d_lens, *d_pos, *d_ids;
    CK(cudaMalloc(&k_pool, pool_elems * sizeof(__half)));
    CK(cudaMalloc(&v_pool, pool_elems * sizeof(__half)));
    CK(cudaMemset(k_pool, 0, pool_elems * sizeof(__half)));
    CK(cudaMemset(v_pool, 0, pool_elems * sizeof(__half)));
    CK(cudaMalloc(&x,      (size_t)cfg.hidden_dim * sizeof(__half)));
    CK(cudaMalloc(&logits, (size_t)cfg.vocab_size * sizeof(__half)));
    CK(cudaMalloc(&d_table, table.size() * sizeof(int)));
    CK(cudaMalloc(&d_lens, sizeof(int)));
    CK(cudaMalloc(&d_pos,  sizeof(int)));
    CK(cudaMalloc(&d_ids,  sizeof(int)));
    CK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(int),
                  cudaMemcpyHostToDevice));

    DecodeScratch scratch;
    scratch.alloc(cfg, 1);

    cublasHandle_t blas;
    cublasCreate(&blas);

    const int H = cfg.hidden_dim;
    // hidden state after each layer, at the LAST token — filled on the final step
    std::vector<std::vector<float>> ours(cfg.num_layers);

    // ── Feed the prompt one token at a time ──────────────────────────────────
    for (int t = 0; t < N; ++t) {
        // Convention: `lens` counts tokens INCLUDING the one being processed.
        // write_kv_paged stores at lens-1; attention scans [0, lens) so the new
        // token can attend to itself.
        const int len = t + 1;
        CK(cudaMemcpy(d_ids,  &tokens[t], sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_lens, &len,       sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos,  &t,         sizeof(int), cudaMemcpyHostToDevice));

        launch_embedding(weights.embed_tokens, d_ids, x, 1, H, 0);

        for (int l = 0; l < cfg.num_layers; ++l) {
            forward_layer(cfg, weights.layers[l], blas, scratch, x,
                          k_pool + (size_t)l * pool_per_layer,
                          v_pool + (size_t)l * pool_per_layer,
                          d_table, d_lens, d_pos,
                          /*B=*/1, BS, max_blocks, /*max_len=*/t + 1, 0);
            // (num_blocks == max_blocks here, which is why the pool-offset bug
            //  in forward_decode did not show up until paged_runtime.)
            if (t == N - 1) {
                CK(cudaDeviceSynchronize());
                ours[l] = download(x, H);       // snapshot after this layer
            }
        }
    }

    launch_rmsnorm(x, weights.final_norm, scratch.h, 1, H, cfg.rms_eps, 0);
    gemm(blas, weights.lm_head, scratch.h, logits, cfg.vocab_size, H, 1);
    CK(cudaDeviceSynchronize());

    // ── Compare ──────────────────────────────────────────────────────────────
    // HF's hidden_states[L+1] is after layer L, for ALL positions; we compare
    // row N-1. NOTE hidden_states[-1] already has model.norm applied, so the
    // last layer file is skipped here and checked via final_norm instead.
    printf("%-14s %12s\n", "after", "rel err");
    printf("---------------------------\n");
    int bad = 0;
    for (int l = 0; l < cfg.num_layers - 1; ++l) {
        char name[32];
        snprintf(name, sizeof name, "layer%02d", l);
        auto ref = read_f32(rdir + "/" + name + ".bin", (size_t)N * H);
        const double e = rel_err(ours[l], ref, (size_t)(N - 1) * H, H);
        printf("%-14s %12.6f%s\n", name, e, e < 2e-2 ? "" : "   <-- FAIL");
        if (e >= 2e-2) { ++bad; break; }         // first bad layer is the bug
    }

    {
        auto ref = read_f32(rdir + "/final_norm.bin", (size_t)N * H);
        auto got = download(scratch.h, H);
        const double e = rel_err(got, ref, (size_t)(N - 1) * H, H);
        printf("%-14s %12.6f%s\n", "final_norm", e, e < 2e-2 ? "" : "   <-- FAIL");
        if (e >= 2e-2) ++bad;
    }
    {
        auto ref = read_f32(rdir + "/logits.bin", (size_t)N * cfg.vocab_size);
        auto got = download(logits, cfg.vocab_size);
        const double e = rel_err(got, ref, (size_t)(N - 1) * cfg.vocab_size,
                                 cfg.vocab_size);
        // argmax agreement matters more than the norm for logits
        int a = 0, b = 0;
        for (int i = 1; i < cfg.vocab_size; ++i) {
            if (got[i] > got[a]) a = i;
            if (ref[(size_t)(N - 1) * cfg.vocab_size + i]
                > ref[(size_t)(N - 1) * cfg.vocab_size + b]) b = i;
        }
        printf("%-14s %12.6f   argmax ours=%d hf=%d %s\n", "logits", e, a, b,
               a == b ? "MATCH" : "MISMATCH");
        if (a != b) ++bad;
    }

    printf("\n%s\n", bad ? "FAILED — first failing layer is where to look"
                         : "all layers match");

    scratch.free();
    weights.free();
    cublasDestroy(blas);
    return bad ? 1 : 0;
}
