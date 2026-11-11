/**
 * test_paged_attention_varlen.cu — S3.1 correctness, standalone.
 *
 * The whole test rests on one equivalence:
 *
 *     varlen with q_len = N  ==  N separate decode calls with the cache
 *                                grown one token at a time
 *
 * If that holds, the mask and the indexing are right, because a decode call at
 * length L is by construction "attend to exactly keys 0..L-1". Running the
 * decode kernel as the oracle also means any bug shared by both kernels
 * (e.g. the GQA head mapping) cancels out — this test isolates exactly what
 * S3.1 changed, nothing else.
 *
 *   1. q_len == 1 everywhere   — varlen must reproduce decode bit-for-bit.
 *                                Isolates the varlen plumbing from the mask:
 *                                here g and b coincide, so an out[b]/out[g]
 *                                mix-up is still invisible.
 *   2. ragged q_len (5, 1, 3)  — g and b now diverge. This is the case that
 *                                catches out[b], and it is also the shape of a
 *                                speculative-decoding verify pass.
 *   3. shuffled physical blocks— same logical sequences, scattered storage,
 *                                must give an identical answer.
 *
 * Build:
 *   nvcc -std=c++17 -I kernels -I src tools/test_paged_attention_varlen.cu \
 *        -o build/test_varlen --extended-lambda -arch=sm_120
 * Run:
 *   ./build/test_varlen
 */

#include "paged_attention_varlen.cuh"
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

namespace {

constexpr int B          = 3;
constexpr int Hq         = 8;
constexpr int Hkv        = 2;
constexpr int D          = 64;
constexpr int BS         = 16;
constexpr int kNumBlocks = 64;

const std::vector<int> kLens = {35, 5, 100};   // cached tokens per sequence

struct Query { int seq; int pos; };            // one entry per query token

// Device-side fixture: KV pool + block table, uploaded once and shared by both
// kernels so that "bit-identical" is a meaningful claim.
struct Fixture {
    __half *k_pool = nullptr, *v_pool = nullptr;
    int    *table  = nullptr;
    int     max_blocks = 0;

    void free_all() { cudaFree(k_pool); cudaFree(v_pool); cudaFree(table); }
};

std::mt19937 rng(0);
std::normal_distribution<float> nd(0.f, 0.5f);

std::vector<float> rnd(size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// host KV, indexed [seq][token][kv_head * D + d]
std::vector<std::vector<std::vector<float>>> g_k(B), g_v(B);

/// Builds the pool + table. `perm` optionally remaps physical block ids.
Fixture make_fixture(const std::vector<int>& host_table, int max_blocks) {
    Fixture f;
    f.max_blocks = max_blocks;

    const size_t pool_elems = (size_t)kNumBlocks * Hkv * BS * D;
    std::vector<__half> h_k(pool_elems, __float2half(0.f)), h_v = h_k;

    for (int b = 0; b < B; ++b)
        for (int t = 0; t < kLens[b]; ++t) {
            const int phys = host_table[(size_t)b * max_blocks + t / BS];
            const int off  = t % BS;
            for (int kvh = 0; kvh < Hkv; ++kvh)
                for (int d = 0; d < D; ++d) {
                    const size_t idx =
                        (((size_t)phys * Hkv + kvh) * BS + off) * D + d;
                    h_k[idx] = __float2half(g_k[b][t][(size_t)kvh * D + d]);
                    h_v[idx] = __float2half(g_v[b][t][(size_t)kvh * D + d]);
                }
        }

    CK(cudaMalloc(&f.k_pool, pool_elems * sizeof(__half)));
    CK(cudaMalloc(&f.v_pool, pool_elems * sizeof(__half)));
    CK(cudaMalloc(&f.table,  host_table.size() * sizeof(int)));
    CK(cudaMemcpy(f.k_pool, h_k.data(), pool_elems*sizeof(__half),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(f.v_pool, h_v.data(), pool_elems*sizeof(__half),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(f.table, host_table.data(), host_table.size()*sizeof(int),
                  cudaMemcpyHostToDevice));
    return f;
}

/// One varlen launch over all query tokens. Returns [total_q * Hq * D] fp32.
std::vector<float> run_varlen(const Fixture& f,
                              const std::vector<Query>& qs,
                              const std::vector<__half>& h_q) {
    const int total_q = (int)qs.size();
    std::vector<int> h_seq(total_q), h_pos(total_q);
    int max_scan = 0;
    for (int g = 0; g < total_q; ++g) {
        h_seq[g] = qs[g].seq;
        h_pos[g] = qs[g].pos;
        max_scan = std::max(max_scan, qs[g].pos + 1);
    }

    __half *d_q, *d_out;
    int *d_seq, *d_pos;
    CK(cudaMalloc(&d_q,   h_q.size() * sizeof(__half)));
    CK(cudaMalloc(&d_out, (size_t)total_q * Hq * D * sizeof(__half)));
    CK(cudaMalloc(&d_seq, total_q * sizeof(int)));
    CK(cudaMalloc(&d_pos, total_q * sizeof(int)));
    CK(cudaMemcpy(d_q, h_q.data(), h_q.size()*sizeof(__half),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_seq, h_seq.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, h_pos.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));

    launch_paged_attention_varlen(d_q, f.k_pool, f.v_pool, f.table,
                                  d_seq, d_pos, d_out,
                                  total_q, Hq, Hkv, D, BS,
                                  f.max_blocks, max_scan,
                                  1.f / std::sqrt((float)D), 0);
    CK(cudaDeviceSynchronize());

    std::vector<__half> h_out((size_t)total_q * Hq * D);
    CK(cudaMemcpy(h_out.data(), d_out, h_out.size()*sizeof(__half),
                  cudaMemcpyDeviceToHost));
    cudaFree(d_q); cudaFree(d_out); cudaFree(d_seq); cudaFree(d_pos);

    std::vector<float> out(h_out.size());
    for (size_t i = 0; i < h_out.size(); ++i) out[i] = __half2float(h_out[i]);
    return out;
}

/// Oracle: the ORIGINAL decode kernel, one query, cache truncated to pos+1.
/// Points at row `seq` of the shared table, so both kernels read the same
/// bytes from the same pool.
std::vector<float> run_decode_one(const Fixture& f, const Query& qu,
                                  const __half* d_q_one) {
    const int n_keys = qu.pos + 1;

    __half* d_out;
    int *d_lens;
    CK(cudaMalloc(&d_out, (size_t)Hq * D * sizeof(__half)));
    CK(cudaMalloc(&d_lens, sizeof(int)));
    CK(cudaMemcpy(d_lens, &n_keys, sizeof(int), cudaMemcpyHostToDevice));

    launch_paged_attention_decode(
        d_q_one, f.k_pool, f.v_pool,
        f.table + (size_t)qu.seq * f.max_blocks,   // this sequence's row
        d_lens, d_out,
        /*B=*/1, Hq, Hkv, D, BS, f.max_blocks, n_keys,
        1.f / std::sqrt((float)D), 0);
    CK(cudaDeviceSynchronize());

    std::vector<__half> h_out((size_t)Hq * D);
    CK(cudaMemcpy(h_out.data(), d_out, h_out.size()*sizeof(__half),
                  cudaMemcpyDeviceToHost));
    cudaFree(d_out); cudaFree(d_lens);

    std::vector<float> out(h_out.size());
    for (size_t i = 0; i < h_out.size(); ++i) out[i] = __half2float(h_out[i]);
    return out;
}

/// Compares a varlen batch against per-query decode calls. Returns max|diff|.
double check_against_decode(const Fixture& f, const std::vector<Query>& qs,
                            const std::vector<__half>& h_q,
                            const std::vector<float>& varlen_out) {
    __half* d_q_one;
    CK(cudaMalloc(&d_q_one, (size_t)Hq * D * sizeof(__half)));

    double worst = 0;
    for (size_t g = 0; g < qs.size(); ++g) {
        CK(cudaMemcpy(d_q_one, h_q.data() + g * Hq * D,
                      (size_t)Hq * D * sizeof(__half), cudaMemcpyHostToDevice));
        const auto ref = run_decode_one(f, qs[g], d_q_one);
        for (size_t i = 0; i < ref.size(); ++i)
            worst = std::max(worst,
                (double)std::fabs(ref[i] - varlen_out[g * Hq * D + i]));
    }
    cudaFree(d_q_one);
    return worst;
}

void report(const char* name, double diff, bool exact) {
    const bool bad = exact ? (diff != 0.0) : (diff > 2e-2);
    printf("   %-34s max|diff| = %.8f  %s\n",
           name, diff, bad ? "FAIL" : "ok");
    if (bad) ++failures;
}

}  // namespace

int main() {
    // ── Fixture: KV for every cached token of every sequence ────────────────
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < kLens[b]; ++t) {
            g_k[b].push_back(rnd((size_t)Hkv * D));
            g_v[b].push_back(rnd((size_t)Hkv * D));
        }

    BlockAllocator alloc;
    alloc.configure({BS, kNumBlocks, 0});
    for (int b = 0; b < B; ++b) alloc.allocate(b + 1, kLens[b]);
    int max_blocks = 0;
    for (int b = 0; b < B; ++b)
        max_blocks = std::max(max_blocks, alloc.blocks_for(kLens[b]));
    std::vector<uint64_t> ids;
    for (int b = 0; b < B; ++b) ids.push_back(b + 1);
    const std::vector<int> table_a = alloc.flatten(ids, max_blocks);

    Fixture fa = make_fixture(table_a, max_blocks);

    auto make_q = [&](const std::vector<Query>& qs) {
        std::vector<__half> h(qs.size() * Hq * D);
        for (size_t i = 0; i < h.size(); ++i) h[i] = __float2half(nd(rng));
        return h;
    };

    // ── 1. q_len == 1 everywhere: must reproduce the decode kernel ──────────
    printf("1. q_len == 1 (varlen must equal decode)\n");
    std::vector<Query> qs1;
    for (int b = 0; b < B; ++b) qs1.push_back({b, kLens[b] - 1});
    const auto hq1 = make_q(qs1);
    const auto out1 = run_varlen(fa, qs1, hq1);
    report("3 sequences, 1 query each", check_against_decode(fa, qs1, hq1, out1),
           /*exact=*/true);

    // ── 2. ragged q_len: g and b diverge ────────────────────────────────────
    // seq 0 contributes 5 query tokens, seq 1 one, seq 2 three. Positions are
    // the LAST q_len positions of each cached sequence, which is exactly what a
    // prefill chunk (or a speculative verify) looks like.
    printf("2. ragged q_len 5 / 1 / 3 (g != b)\n");
    std::vector<Query> qs2;
    const int qlen[B] = {5, 1, 3};
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < qlen[b]; ++i)
            qs2.push_back({b, kLens[b] - qlen[b] + i});
    const auto hq2 = make_q(qs2);
    const auto out2 = run_varlen(fa, qs2, hq2);
    printf("   total_q = %zu across %d sequences\n", qs2.size(), B);
    report("each query vs its own decode call",
           check_against_decode(fa, qs2, hq2, out2), /*exact=*/true);

    // ── 3. shuffled physical blocks ─────────────────────────────────────────
    printf("3. shuffled physical blocks\n");
    std::vector<int> perm(kNumBlocks);
    for (int i = 0; i < kNumBlocks; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> table_b = table_a;
    for (auto& x : table_b) if (x >= 0) x = perm[x];

    Fixture fb = make_fixture(table_b, max_blocks);
    const auto out2_shuf = run_varlen(fb, qs2, hq2);
    double diff_shuf = 0;
    for (size_t i = 0; i < out2.size(); ++i)
        diff_shuf = std::max(diff_shuf,
                             (double)std::fabs(out2[i] - out2_shuf[i]));
    report("same answer, scattered storage", diff_shuf, /*exact=*/true);

    fa.free_all();
    fb.free_all();
    printf("\n%s\n", failures ? "FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
