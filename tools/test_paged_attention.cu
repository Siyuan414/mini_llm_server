/**
 * test_paged_attention.cu — S2.8 correctness, standalone (no TRT, no engine).
 *
 * Three checks:
 *   1. vs CPU reference       — the maths is right
 *   2. shuffled block table   — SAME answer when the identical logical sequence
 *                               is stored in scattered physical blocks. This is
 *                               the property paging depends on; if it fails,
 *                               the indirection is wrong.
 *   3. ragged batch           — sequences of different lengths in one launch do
 *                               not contaminate each other
 *
 * Build:
 *   nvcc -std=c++17 -I kernels -I src tools/test_paged_attention.cu \
 *        -o build/test_paged --extended-lambda -arch=sm_120
 * Run:
 *   ./build/test_paged
 */

#include "paged_attention.cuh"
#include "block_allocator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                   \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    exit(1);} } while(0)

static int failures = 0;

// ── CPU reference: plain contiguous attention, fp32 ─────────────────────────
static std::vector<float> cpu_reference(
        const std::vector<float>& q,                       // [Hq, D]
        const std::vector<std::vector<float>>& k,          // [len][Hkv*D]
        const std::vector<std::vector<float>>& v,
        int Hq, int Hkv, int D, float scale) {
    const int len = (int)k.size();
    std::vector<float> out((size_t)Hq * D, 0.f);
    for (int h = 0; h < Hq; ++h) {
        const int kvh = h / (Hq / Hkv);
        std::vector<float> s(len);
        for (int t = 0; t < len; ++t) {
            float acc = 0.f;
            for (int d = 0; d < D; ++d)
                acc += q[(size_t)h * D + d] * k[t][(size_t)kvh * D + d];
            s[t] = acc * scale;
        }
        const float m = *std::max_element(s.begin(), s.end());
        float sum = 0.f;
        for (float& x : s) { x = std::exp(x - m); sum += x; }
        for (int d = 0; d < D; ++d) {
            float acc = 0.f;
            for (int t = 0; t < len; ++t) acc += s[t] * v[t][(size_t)kvh * D + d];
            out[(size_t)h * D + d] = acc / sum;
        }
    }
    return out;
}

struct Case {
    int B, Hq, Hkv, D, BS, num_blocks;
    std::vector<int> lens;
};

// Runs the kernel with a caller-supplied block table so the same logical data
// can be placed in different physical blocks.
static std::vector<float> run_kernel(
        const Case& c,
        const std::vector<std::vector<float>>& q,                 // [B][Hq*D]
        const std::vector<std::vector<std::vector<float>>>& k,    // [B][len][Hkv*D]
        const std::vector<std::vector<std::vector<float>>>& v,
        const std::vector<int>& table, int max_blocks) {

    const int B = c.B, Hq = c.Hq, Hkv = c.Hkv, D = c.D, BS = c.BS;
    const size_t pool_elems = (size_t)c.num_blocks * Hkv * BS * D;

    // Scatter each sequence's KV into its assigned physical blocks.
    std::vector<__half> h_k(pool_elems, __float2half(0.f)), h_v = h_k;
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < c.lens[b]; ++t) {
            const int phys = table[(size_t)b * max_blocks + t / BS];
            const int off  = t % BS;
            for (int kvh = 0; kvh < Hkv; ++kvh)
                for (int d = 0; d < D; ++d) {
                    const size_t idx = (((size_t)phys * Hkv + kvh) * BS + off) * D + d;
                    h_k[idx] = __float2half(k[b][t][(size_t)kvh * D + d]);
                    h_v[idx] = __float2half(v[b][t][(size_t)kvh * D + d]);
                }
        }

    std::vector<__half> h_q((size_t)B * Hq * D);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < Hq * D; ++i)
            h_q[(size_t)b * Hq * D + i] = __float2half(q[b][i]);

    __half *d_q, *d_k, *d_v, *d_out;
    int *d_tab, *d_lens;
    CK(cudaMalloc(&d_q,   h_q.size() * sizeof(__half)));
    CK(cudaMalloc(&d_k,   pool_elems * sizeof(__half)));
    CK(cudaMalloc(&d_v,   pool_elems * sizeof(__half)));
    CK(cudaMalloc(&d_out, (size_t)B * Hq * D * sizeof(__half)));
    CK(cudaMalloc(&d_tab, table.size() * sizeof(int)));
    CK(cudaMalloc(&d_lens, c.lens.size() * sizeof(int)));
    CK(cudaMemcpy(d_q, h_q.data(), h_q.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_k, h_k.data(), pool_elems*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v, h_v.data(), pool_elems*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tab, table.data(), table.size()*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lens, c.lens.data(), c.lens.size()*sizeof(int), cudaMemcpyHostToDevice));

    const int max_len = *std::max_element(c.lens.begin(), c.lens.end());
    launch_paged_attention_decode(d_q, d_k, d_v, d_tab, d_lens, d_out,
                                  B, Hq, Hkv, D, BS, max_blocks, max_len,
                                  1.f / std::sqrt((float)D), 0);
    CK(cudaDeviceSynchronize());

    std::vector<__half> h_out((size_t)B * Hq * D);
    CK(cudaMemcpy(h_out.data(), d_out, h_out.size()*sizeof(__half),
                  cudaMemcpyDeviceToHost));
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_out); cudaFree(d_tab); cudaFree(d_lens);

    std::vector<float> out(h_out.size());
    for (size_t i = 0; i < h_out.size(); ++i) out[i] = __half2float(h_out[i]);
    return out;
}

int main() {
    Case c{/*B*/3, /*Hq*/8, /*Hkv*/2, /*D*/64, /*BS*/16, /*num_blocks*/64,
           /*lens*/{35, 5, 100}};   // ragged on purpose: mid-block, tiny, long

    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 0.5f);
    auto rnd = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng);
        return v;
    };

    std::vector<std::vector<float>> q(c.B);
    std::vector<std::vector<std::vector<float>>> k(c.B), v(c.B);
    for (int b = 0; b < c.B; ++b) {
        q[b] = rnd((size_t)c.Hq * c.D);
        for (int t = 0; t < c.lens[b]; ++t) {
            k[b].push_back(rnd((size_t)c.Hkv * c.D));
            v[b].push_back(rnd((size_t)c.Hkv * c.D));
        }
    }

    // ── Table 1: sequential blocks via the real allocator ───────────────────
    BlockAllocator alloc;
    alloc.configure({c.BS, c.num_blocks, 0});
    for (int b = 0; b < c.B; ++b) alloc.allocate(b + 1, c.lens[b]);
    int max_blocks = 0;
    for (int b = 0; b < c.B; ++b)
        max_blocks = std::max(max_blocks, alloc.blocks_for(c.lens[b]));
    std::vector<uint64_t> ids;
    for (int b = 0; b < c.B; ++b) ids.push_back(b + 1);
    std::vector<int> table_a = alloc.flatten(ids, max_blocks);

    // ── Table 2: same logical sequences, SHUFFLED physical blocks ───────────
    std::vector<int> perm(c.num_blocks);
    for (int i = 0; i < c.num_blocks; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> table_b = table_a;
    for (auto& x : table_b) if (x >= 0) x = perm[x];

    const float scale = 1.f / std::sqrt((float)c.D);
    auto out_a = run_kernel(c, q, k, v, table_a, max_blocks);
    auto out_b = run_kernel(c, q, k, v, table_b, max_blocks);

    // ── 1. vs CPU reference ─────────────────────────────────────────────────
    printf("1. vs CPU reference (fp32)\n");
    double worst = 0;
    for (int b = 0; b < c.B; ++b) {
        auto ref = cpu_reference(q[b], k[b], v[b], c.Hq, c.Hkv, c.D, scale);
        double d = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            d = std::max(d, (double)std::fabs(ref[i] - out_a[(size_t)b*c.Hq*c.D + i]));
        worst = std::max(worst, d);
        printf("   seq %d (len %3d)  max|diff| = %.5f\n", b, c.lens[b], d);
    }
    if (worst > 2e-2) { printf("   FAIL: exceeds fp16 tolerance\n"); ++failures; }

    // ── 2. block placement must not change the answer ───────────────────────
    printf("2. shuffled physical blocks\n");
    double diff_shuffle = 0;
    for (size_t i = 0; i < out_a.size(); ++i)
        diff_shuffle = std::max(diff_shuffle, (double)std::fabs(out_a[i] - out_b[i]));
    printf("   max|diff| = %.8f  (must be exactly 0)\n", diff_shuffle);
    if (diff_shuffle != 0.0) {
        printf("   FAIL: output depends on WHERE blocks live — indirection is wrong\n");
        ++failures;
    }

    // ── 3. ragged batch isolation ───────────────────────────────────────────
    printf("3. ragged batch isolation\n");
    Case solo = c; solo.B = 1; solo.lens = {c.lens[1]};
    std::vector<uint64_t> id1{1};
    BlockAllocator a2; a2.configure({c.BS, c.num_blocks, 0});
    a2.allocate(1, c.lens[1]);
    auto table_solo = a2.flatten(id1, max_blocks);
    auto out_solo = run_kernel(solo, {q[1]}, {k[1]}, {v[1]}, table_solo, max_blocks);
    double d_solo = 0;
    for (size_t i = 0; i < out_solo.size(); ++i)
        d_solo = std::max(d_solo,
                          (double)std::fabs(out_solo[i] - out_a[(size_t)1*c.Hq*c.D + i]));
    printf("   seq 1 alone vs in ragged batch: max|diff| = %.8f\n", d_solo);
    if (d_solo != 0.0) {
        printf("   FAIL: neighbouring sequences are contaminating each other\n");
        ++failures;
    }

    printf("\n%s\n", failures ? "FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
