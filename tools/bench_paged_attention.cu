/**
 * bench_paged_attention.cu — S2.8: paged vs fixed-slot attention.
 *
 * Answers the question Stage 2A left open: efficiency fell to 32% at batch 16
 * because every slot scanned the full max_seq window regardless of its real
 * length. How much of that does paging give back?
 *
 * The comparison is between two kernels doing the SAME maths:
 *   paged      — reads exactly lens[b] tokens, KV scattered across blocks
 *   fixed slot — reads max_seq tokens for every sequence, masking the tail
 *                (this is what TRT attention does in Stage 2A)
 *
 * Build:
 *   nvcc -std=c++17 -I kernels -I src tools/bench_paged_attention.cu \
 *        -o build/bench_paged --extended-lambda -arch=sm_120 -O3
 */

#include "paged_attention.cuh"
#include "block_allocator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                   \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    exit(1);} } while(0)

// ── Fixed-slot attention: what Stage 2A actually runs ───────────────────────
// KV is [B, Hkv, max_seq, D]; every sequence scans all max_seq positions and
// masks the tail. Identical arithmetic to the paged kernel, different addressing
// and a different amount of work.
__global__ void fixed_slot_attention_decode(
        const __half* __restrict__ q,
        const __half* __restrict__ k,        // [B, Hkv, max_seq, D]
        const __half* __restrict__ v,
        const int*    __restrict__ lens,
        __half*       __restrict__ out,
        int Hq, int Hkv, int D, int max_seq, float scale) {

    const int h = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const int real_len = lens[b];
    const int kv_head  = h / (Hq / Hkv);
    const __half* qv   = q + ((size_t)b * Hq + h) * D;

    extern __shared__ float smem[];
    float* scores = smem;                       // [max_seq] — the whole window
    float* reduce = smem + max_seq;

    // Scans the FULL window; positions past real_len are masked out but still
    // cost a memory read and a dot product.
    for (int t = tid; t < max_seq; t += blockDim.x) {
        if (t >= real_len) { scores[t] = -FLT_MAX; continue; }
        const __half* kv = k + (((size_t)b * Hkv + kv_head) * max_seq + t) * D;
        float acc = 0.f;
        for (int d = 0; d < D; ++d) acc += __half2float(qv[d]) * __half2float(kv[d]);
        scores[t] = acc * scale;
    }
    __syncthreads();

    float lm = -FLT_MAX;
    for (int t = tid; t < max_seq; t += blockDim.x) lm = fmaxf(lm, scores[t]);
    const float m = paged_attn::block_reduce(lm, reduce,
        [] __device__ (float a, float c) { return fmaxf(a, c); });

    float ls = 0.f;
    for (int t = tid; t < max_seq; t += blockDim.x) {
        const float e = (scores[t] == -FLT_MAX) ? 0.f : __expf(scores[t] - m);
        scores[t] = e; ls += e;
    }
    __syncthreads();
    const float l = paged_attn::block_reduce(ls, reduce,
        [] __device__ (float a, float c) { return a + c; });
    const float inv_l = 1.f / l;

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < real_len; ++t) {
            const __half* vv = v + (((size_t)b * Hkv + kv_head) * max_seq + t) * D;
            acc += scores[t] * __half2float(vv[d]);
        }
        out[((size_t)b * Hq + h) * D + d] = __float2half(acc * inv_l);
    }
}

struct Result { double us; double mem_mb; double kv_read_mb; };

static double time_kernel(std::function<void()> launch, int iters = 200) {
    for (int i = 0; i < 20; ++i) launch();          // warmup
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1000.0 / iters;                     // microseconds per call
}

int main(int argc, char** argv) {
    const int Hq = 32, Hkv = 8, D = 64, BS = 16;
    const int max_seq = (argc > 1) ? atoi(argv[1]) : 512;
    const float scale = 1.f / std::sqrt((float)D);

    printf("paged vs fixed-slot decode attention  (Hq=%d Hkv=%d D=%d "
           "block=%d max_seq=%d)\n\n", Hq, Hkv, D, BS, max_seq);
    printf("%5s %9s %11s %11s %8s   %10s %10s %7s\n",
           "B", "mean len", "paged us", "fixed us", "speedup",
           "paged MB", "fixed MB", "saved");
    printf("%s\n", std::string(84, '-').c_str());

    std::mt19937 rng(1234);

    for (int B : {4, 8, 16, 32, 64}) {
        // Realistic ragged workload: most requests short, a few long — the
        // shape that makes fixed slots wasteful.
        std::vector<int> lens(B);
        std::lognormal_distribution<double> ln(std::log(50.0), 0.6);
        for (int& L : lens)
            L = std::max(1, std::min(max_seq, (int)ln(rng)));
        const double mean_len =
            std::accumulate(lens.begin(), lens.end(), 0.0) / B;

        // ── paged layout ────────────────────────────────────────────────────
        BlockAllocator alloc;
        const int pool_blocks = B * (max_seq / BS) + 64;
        alloc.configure({BS, pool_blocks, 0});
        std::vector<uint64_t> ids;
        int max_blocks = 0;
        for (int b = 0; b < B; ++b) {
            alloc.allocate(b + 1, lens[b]);
            ids.push_back(b + 1);
            max_blocks = std::max(max_blocks, alloc.blocks_for(lens[b]));
        }
        // Shuffle physical placement so the benchmark reflects a fragmented
        // pool, not a conveniently sequential one.
        std::vector<int> table = alloc.flatten(ids, max_blocks);
        int used_blocks = 0;
        for (int x : table) if (x >= 0) ++used_blocks;

        const size_t pool_elems = (size_t)pool_blocks * Hkv * BS * D;
        const size_t fixed_elems = (size_t)B * Hkv * max_seq * D;

        __half *d_q, *d_kp, *d_vp, *d_kf, *d_vf, *d_out;
        int *d_tab, *d_lens;
        CK(cudaMalloc(&d_q,  (size_t)B * Hq * D * sizeof(__half)));
        CK(cudaMalloc(&d_out,(size_t)B * Hq * D * sizeof(__half)));
        CK(cudaMalloc(&d_kp, pool_elems  * sizeof(__half)));
        CK(cudaMalloc(&d_vp, pool_elems  * sizeof(__half)));
        CK(cudaMalloc(&d_kf, fixed_elems * sizeof(__half)));
        CK(cudaMalloc(&d_vf, fixed_elems * sizeof(__half)));
        CK(cudaMalloc(&d_tab, table.size() * sizeof(int)));
        CK(cudaMalloc(&d_lens, B * sizeof(int)));
        CK(cudaMemset(d_kp, 0x11, pool_elems * sizeof(__half)));
        CK(cudaMemset(d_vp, 0x11, pool_elems * sizeof(__half)));
        CK(cudaMemset(d_kf, 0x11, fixed_elems * sizeof(__half)));
        CK(cudaMemset(d_vf, 0x11, fixed_elems * sizeof(__half)));
        CK(cudaMemset(d_q, 0x11, (size_t)B * Hq * D * sizeof(__half)));
        CK(cudaMemcpy(d_tab, table.data(), table.size()*sizeof(int),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_lens, lens.data(), B*sizeof(int), cudaMemcpyHostToDevice));

        const int max_len = *std::max_element(lens.begin(), lens.end());

        const double us_paged = time_kernel([&]{
            launch_paged_attention_decode(d_q, d_kp, d_vp, d_tab, d_lens, d_out,
                                          B, Hq, Hkv, D, BS, max_blocks,
                                          max_len, scale, 0);
        });

        const double us_fixed = time_kernel([&]{
            const dim3 grid(Hq, B);
            const size_t smem = (size_t)(max_seq + paged_attn::kThreads) * sizeof(float);
            fixed_slot_attention_decode<<<grid, paged_attn::kThreads, smem>>>(
                d_q, d_kf, d_vf, d_lens, d_out, Hq, Hkv, D, max_seq, scale);
        });

        // Memory actually holding this batch's KV (not the whole pool).
        const double paged_mb = (double)used_blocks * Hkv * BS * D * 2 * 2 / 1e6;
        const double fixed_mb = (double)B * Hkv * max_seq * D * 2 * 2 / 1e6;

        printf("%5d %9.1f %11.1f %11.1f %7.2fx   %10.1f %10.1f %6.1fx\n",
               B, mean_len, us_paged, us_fixed, us_fixed / us_paged,
               paged_mb, fixed_mb, fixed_mb / paged_mb);

        cudaFree(d_q); cudaFree(d_out); cudaFree(d_kp); cudaFree(d_vp);
        cudaFree(d_kf); cudaFree(d_vf); cudaFree(d_tab); cudaFree(d_lens);
    }

    printf("\nMB counts K and V across all layers?  No — one layer. Multiply by\n"
           "num_layers (16 for LLaMA-3.2-1B) for the whole-model figure.\n");
    return 0;
}
