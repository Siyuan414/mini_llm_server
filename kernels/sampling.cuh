/**
 * sampling.cuh — GPU sampler: temperature + top-k + top-p (nucleus)
 *
 * Everything stays on device. Only the 4-byte sampled token id crosses back to
 * the host, same contract as GpuArgmax — no 250 KB logits transfer per step.
 *
 * Pipeline (all on one stream):
 *   1. k_scale   : fp16 logits → fp32, divided by temperature
 *   2. RadixSort : SortPairsDescending(scaled, iota) → sorted values + token ids
 *   3. k_exp     : p[i] = exp(sorted[i] - sorted[0])   (sorted[0] is the max,
 *                  so this is the numerically-stable softmax numerator)
 *   4. InclusiveSum → cumulative probability mass
 *   5. k_sample  : one thread — binary-search the top-k/top-p cutoff, then
 *                  binary-search the sampled position. 2 x ~17 steps.
 *
 * Filtering order matches HuggingFace: top-k first, then top-p renormalized over
 * the surviving k tokens.
 *
 * A full 128k-element sort per token is more work than strictly necessary — a
 * radix top-k would avoid sorting the tail. It is measured in S1.6; at ~1B-model
 * decode latency it is not the bottleneck, and correctness-first is the right
 * default before optimizing.
 */

#pragma once

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <stdexcept>

namespace sampling_kernels {

__global__ void k_scale(const __half* logits, float* out, int n, float inv_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(logits[i]) * inv_t;
}

__global__ void k_iota(int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) idx[i] = i;
}

// sorted is descending, so sorted[0] is the max → stable softmax numerator.
__global__ void k_exp(const float* sorted, float* probs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) probs[i] = __expf(sorted[i] - sorted[0]);
}

// Single-thread: cutoff search + inverse-CDF sample. Both are binary searches
// over the cumulative mass, so ~17 steps each for a 128k vocab.
__global__ void k_sample(const float* cumsum, const int* sorted_idx, int n,
                         int top_k, float top_p, float r, int* out) {
    if (threadIdx.x || blockIdx.x) return;

    const int kmax = (top_k > 0 && top_k < n) ? top_k : n;

    // top-p over the top-k survivors (renormalized — matches HF ordering)
    const float total  = cumsum[kmax - 1];
    const float target = top_p * total;

    int lo = 0, hi = kmax - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cumsum[mid] >= target) hi = mid; else lo = mid + 1;
    }
    const int cutoff = lo;                 // last token inside the nucleus

    // inverse CDF over [0, cutoff]
    const float pick = r * cumsum[cutoff];
    lo = 0; hi = cutoff;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cumsum[mid] >= pick) hi = mid; else lo = mid + 1;
    }
    *out = sorted_idx[lo];
}

}  // namespace sampling_kernels

struct SamplingParams {
    float temperature = 1.0f;   // <= 0 → greedy (argmax)
    int   top_k       = 50;     // <= 0 → disabled
    float top_p       = 0.95f;  // >= 1 → disabled
    uint64_t seed     = 42;

    bool greedy() const { return temperature <= 0.0f; }
};

class GpuSampler {
public:
    void alloc(int vocab, uint64_t seed = 42) {
        n_ = vocab;
        rng_.seed(seed);

        ck(cudaMalloc(&d_scaled_,     (size_t)n_ * sizeof(float)));
        ck(cudaMalloc(&d_sorted_val_, (size_t)n_ * sizeof(float)));
        ck(cudaMalloc(&d_idx_,        (size_t)n_ * sizeof(int)));
        ck(cudaMalloc(&d_sorted_idx_, (size_t)n_ * sizeof(int)));
        ck(cudaMalloc(&d_probs_,      (size_t)n_ * sizeof(float)));
        ck(cudaMalloc(&d_cumsum_,     (size_t)n_ * sizeof(float)));
        ck(cudaMalloc(&d_out_,        sizeof(int)));

        // Temp storage must satisfy both CUB algorithms — take the max.
        size_t sort_bytes = 0, scan_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, sort_bytes, d_scaled_, d_sorted_val_,
            d_idx_, d_sorted_idx_, n_);
        cub::DeviceScan::InclusiveSum(
            nullptr, scan_bytes, d_probs_, d_cumsum_, n_);
        temp_bytes_ = sort_bytes > scan_bytes ? sort_bytes : scan_bytes;
        ck(cudaMalloc(&d_temp_, temp_bytes_));
    }

    void free() {
        for (void* p : {(void*)d_scaled_, (void*)d_sorted_val_, (void*)d_idx_,
                        (void*)d_sorted_idx_, (void*)d_probs_, (void*)d_cumsum_,
                        (void*)d_out_, d_temp_})
            cudaFree(p);
    }

    size_t bytes() const {
        return (size_t)n_ * (4 * 4 + 2 * 4) + temp_bytes_;
    }

    // Returns the sampled token id. d_logits: [vocab] fp16 on device.
    int sample(const __half* d_logits, const SamplingParams& p,
               cudaStream_t stream) {
        using namespace sampling_kernels;
        const int threads = 256;
        const int blocks  = (n_ + threads - 1) / threads;

        const float inv_t = 1.0f / p.temperature;
        k_scale<<<blocks, threads, 0, stream>>>(d_logits, d_scaled_, n_, inv_t);
        k_iota <<<blocks, threads, 0, stream>>>(d_idx_, n_);

        size_t tb = temp_bytes_;
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_, tb, d_scaled_, d_sorted_val_,
            d_idx_, d_sorted_idx_, n_, 0, sizeof(float) * 8, stream);

        k_exp<<<blocks, threads, 0, stream>>>(d_sorted_val_, d_probs_, n_);

        tb = temp_bytes_;
        cub::DeviceScan::InclusiveSum(
            d_temp_, tb, d_probs_, d_cumsum_, n_, stream);

        const float r = uniform_();
        const float top_p = (p.top_p >= 1.0f || p.top_p <= 0.0f) ? 1.0f : p.top_p;
        k_sample<<<1, 1, 0, stream>>>(d_cumsum_, d_sorted_idx_, n_,
                                      p.top_k, top_p, r, d_out_);

        int h_out = 0;
        ck(cudaMemcpyAsync(&h_out, d_out_, sizeof(int),
                           cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));
        return h_out;
    }

private:
    static void ck(cudaError_t e) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("GpuSampler CUDA: ")
                                     + cudaGetErrorString(e));
    }

    // (0,1] — never exactly 0, so the inverse-CDF search can't underflow to a
    // zero-probability token.
    float uniform_() {
        float u = dist_(rng_);
        return u <= 0.0f ? 1e-7f : u;
    }

    int    n_ = 0;
    float* d_scaled_     = nullptr;
    float* d_sorted_val_ = nullptr;
    int*   d_idx_        = nullptr;
    int*   d_sorted_idx_ = nullptr;
    float* d_probs_      = nullptr;
    float* d_cumsum_     = nullptr;
    int*   d_out_        = nullptr;
    void*  d_temp_       = nullptr;
    size_t temp_bytes_   = 0;

    std::mt19937_64                       rng_;
    std::uniform_real_distribution<float> dist_{0.0f, 1.0f};
};
