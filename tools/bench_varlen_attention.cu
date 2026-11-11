/**
 * bench_varlen_attention.cu — where does the attention kernel's time go? (S4.0)
 *
 * Measure before optimizing. Two questions this answers, and both gate whether
 * S4.1 (vectorized loads) and S4.2 (tiling) are worth writing:
 *
 *   1. us/call across the shapes the server actually runs
 *   2. achieved HBM bandwidth vs the device peak
 *
 * (2) is the decisive one. The kernel reads K and V once per QUERY BLOCK, so a
 * 256-token chunk streams the whole KV cache 256 times. If achieved bandwidth is
 * near peak, the kernel is memory-bound and tiling (loading a K/V tile once into
 * shared memory and reusing it across a tile of queries) is the fix. If it is far
 * below peak, something else is the limit — occupancy, integer work, latency —
 * and tiling would be wasted effort.
 *
 * Bytes moved per call, counting only what the kernel must read:
 *     total_q * Hq * D * 2                      queries
 *   + sum over query tokens of 2 * Hkv * n_keys * D * 2   K and V, re-read
 *   + total_q * Hq * D * 2                      output
 *
 * Note the middle term scales with total_q * n_keys. That product is exactly what
 * tiling attacks.
 *
 * Build:
 *   nvcc -std=c++17 -O3 -I kernels -I src tools/bench_varlen_attention.cu \
 *        -o build/bench_varlen --extended-lambda -arch=sm_120
 * Run:
 *   ./build/bench_varlen
 */

#include "paged_attention_varlen.cuh"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                   \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    exit(1);} } while(0)

namespace {

// LLaMA-3.2-1B
constexpr int Hq  = 32;
constexpr int Hkv = 8;
constexpr int D   = 64;
constexpr int BS  = 16;

struct Shape {
    const char* label;
    int n_seqs;     // sequences in the batch
    int q_len;      // query tokens per sequence
    int context;    // cached tokens per sequence (before this batch)
};

double bench_one(const Shape& sh, double peak_gbs, int iters = 200) {
    const int total_q    = sh.n_seqs * sh.q_len;
    const int max_len    = sh.context + sh.q_len;
    const int max_blocks = (max_len + BS - 1) / BS;
    const int num_blocks = sh.n_seqs * max_blocks;

    // Host-side q_seq / q_pos: each sequence contributes q_len tokens at the
    // end of its context, exactly as the engine builds them.
    std::vector<int> h_seq(total_q), h_pos(total_q);
    int g = 0, max_scan = 0;
    for (int s = 0; s < sh.n_seqs; ++s)
        for (int i = 0; i < sh.q_len; ++i, ++g) {
            h_seq[g] = s;
            h_pos[g] = sh.context + i;
            max_scan = std::max(max_scan, h_pos[g] + 1);
        }

    // Sequential block table — placement does not affect timing materially
    // (verified bit-identical under shuffling in the correctness tests), and
    // sequential keeps the fixture simple.
    std::vector<int> h_tab((size_t)sh.n_seqs * max_blocks);
    for (int s = 0; s < sh.n_seqs; ++s)
        for (int b = 0; b < max_blocks; ++b)
            h_tab[(size_t)s * max_blocks + b] = s * max_blocks + b;

    const size_t pool_elems = (size_t)num_blocks * Hkv * BS * D;

    __half *d_q, *d_k, *d_v, *d_out;
    int *d_seq, *d_pos, *d_tab;
    CK(cudaMalloc(&d_q,   (size_t)total_q * Hq * D * sizeof(__half)));
    CK(cudaMalloc(&d_out, (size_t)total_q * Hq * D * sizeof(__half)));
    CK(cudaMalloc(&d_k,   pool_elems * sizeof(__half)));
    CK(cudaMalloc(&d_v,   pool_elems * sizeof(__half)));
    CK(cudaMalloc(&d_seq, total_q * sizeof(int)));
    CK(cudaMalloc(&d_pos, total_q * sizeof(int)));
    CK(cudaMalloc(&d_tab, h_tab.size() * sizeof(int)));
    CK(cudaMemset(d_q, 0x3C, (size_t)total_q * Hq * D * sizeof(__half)));  // ~1.0
    CK(cudaMemset(d_k, 0x3C, pool_elems * sizeof(__half)));
    CK(cudaMemset(d_v, 0x3C, pool_elems * sizeof(__half)));
    CK(cudaMemcpy(d_seq, h_seq.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, h_pos.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tab, h_tab.data(), h_tab.size()*sizeof(int),
                  cudaMemcpyHostToDevice));

    const float scale = 1.f / std::sqrt((float)D);
    auto launch = [&] {
        launch_paged_attention_varlen(d_q, d_k, d_v, d_tab, d_seq, d_pos, d_out,
                                      total_q, Hq, Hkv, D, BS,
                                      max_blocks, max_scan, scale, 0);
    };

    for (int i = 0; i < 20; ++i) launch();       // warm up
    CK(cudaDeviceSynchronize());

    // MIN of several trials, not a single mean. A mean folds in whatever else
    // the GPU was doing; the minimum is the trial least polluted by thermal
    // throttling and background contention, and it is the reproducible number.
    // Learned the hard way: the same binary measured 113 us on an idle GPU and
    // 179 us with a game open — 58% apart, which would swamp any optimization
    // worth arguing about. The spread below is the honest error bar.
    constexpr int kTrials = 5;
    std::vector<double> trials;
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    for (int t = 0; t < kTrials; ++t) {
        // Let the card recover its boost clock between trials. Without this the
        // spread is 20-30% on a consumer GPU purely from thermal droop — trial 1
        // fast, trial 5 slow — which looks like measurement noise but is not.
        // The MIN is reproducible to ~1% across sessions either way; the sleep
        // just makes the reported spread mean "contention" rather than "heat".
        if (t) { CK(cudaDeviceSynchronize()); usleep(200000); }
        CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) launch();
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
        trials.push_back((double)ms * 1000.0 / iters);
    }
    std::sort(trials.begin(), trials.end());
    const double us     = trials.front();                 // min
    const double median = trials[kTrials / 2];
    const double spread = 100.0 * (trials.back() - us) / us;

    // Bytes the kernel must read, with K/V counted once PER QUERY TOKEN —
    // which is what it actually does, and the term tiling removes.
    double kv_bytes = 0;
    for (int i = 0; i < total_q; ++i)
        kv_bytes += 2.0 * Hkv * (h_pos[i] + 1) * D * sizeof(__half);
    const double qo_bytes = 2.0 * total_q * Hq * D * sizeof(__half);
    const double gbs = (kv_bytes + qo_bytes) / (us * 1e-6) / 1e9;

    printf("  %-22s total_q %5d  ctx %4d  %8.1f us  %7.1f GB/s  %5.1f%% peak"
           "   (med %.1f, spread %.0f%%)%s\n",
           sh.label, total_q, sh.context, us, gbs, 100.0 * gbs / peak_gbs,
           median, spread, spread > 10.0 ? "  <-- NOISY" : "");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    cudaFree(d_seq); cudaFree(d_pos); cudaFree(d_tab);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return us;
}

}  // namespace

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    // memoryClockRate is in kHz, busWidth in bits, DDR -> x2.
    const double peak_gbs =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    printf("%s — %d SMs, %.0f GB/s peak HBM\n\n",
           prop.name, prop.multiProcessorCount, peak_gbs);

    printf("decode steps (q_len = 1)\n");
    const Shape decode[] = {
        {"batch 32, ctx 256",   32, 1,  256},
        {"batch 32, ctx 1024",  32, 1, 1024},
        {"batch 32, ctx 2048",  32, 1, 2048},
        {"batch 64, ctx 1024",  64, 1, 1024},
    };
    for (const auto& s : decode) bench_one(s, peak_gbs);

    printf("\nprefill chunks (q_len > 1) — K/V re-read per query token\n");
    const Shape prefill[] = {
        {"1 x 256 tok, ctx 0",    1, 256,    0},
        {"1 x 256 tok, ctx 256",  1, 256,  256},
        {"1 x 512 tok, ctx 0",    1, 512,    0},
        {"2 x 256 tok, ctx 0",    2, 256,    0},
        {"1 x 128 tok, ctx 1024", 1, 128, 1024},
    };
    for (const auto& s : prefill) bench_one(s, peak_gbs);

    printf("\nmixed step, as the server runs it\n");
    const Shape mixed[] = {
        {"30 decode + 1x256",    1, 256,  512},   // chunk component
        {"30 decode @ ctx 512",  30, 1,   512},   // decode component
    };
    for (const auto& s : mixed) bench_one(s, peak_gbs);

    printf("\nReading this:\n");
    printf("  near peak  -> memory-bound; S4.2 (tiling) is the fix\n");
    printf("  far below  -> occupancy / integer work / latency bound;\n");
    printf("                S4.1 (vectorized loads) first, and profile with Nsight\n");
    return 0;
}
