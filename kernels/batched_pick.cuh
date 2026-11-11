/**
 * batched_pick.cuh — pick a token for all B slots in one shot (Stage 2A, S2.3).
 *
 * ── What this replaces ───────────────────────────────────────────────────────
 * The naive decode step did, per slot:
 *     cublasHgemm(n=1)  →  cub::DeviceReduce::ArgMax  →  cudaStreamSynchronize
 * so a batch of 4 paid FOUR full host round-trips per decode step. Since decode
 * is latency-bound, those syncs were a large share of the step.
 *
 * ── The GEMM needs no batching API at all ────────────────────────────────────
 * cuBLAS is column-major. With ldc = V, C(v,b) lives at v + b*V — which is
 * exactly where row-major [B, V] puts (b, v). Likewise d_hidden as column-major
 * H x B with ldb = H matches row-major [B, H]. So the ONLY change from the
 * single-row version is n = 1 → n = B; no strided-batch call, no relayout.
 *
 * ── Segmented argmax ─────────────────────────────────────────────────────────
 * cub::DeviceSegmentedReduce::ArgMax treats one array as B segments of V and
 * emits B results in a single launch. The returned `key` is the index WITHIN
 * its segment, which is already the token id.
 *
 * Sampling (top-k/top-p) for B rows needs a segmented radix sort plus a
 * per-segment scan; not implemented yet — greedy is the default path and the
 * one the benchmarks use. sample_batched() falls back to a per-row loop.
 */

#pragma once

#include "sampling.cuh"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

class BatchedPicker {
public:
    using Pair = cub::KeyValuePair<int, __half>;

    void alloc(int vocab, int max_batch) {
        v_ = vocab;
        max_b_ = max_batch;

        ck(cudaMalloc(&d_out_, (size_t)max_b_ * sizeof(Pair)));
        h_out_.resize(max_b_);

        // Segment boundaries: [0, V, 2V, ...]. begin = offsets, end = offsets+1,
        // which is why there are max_b_ + 1 of them.
        std::vector<int> off(max_b_ + 1);
        for (int i = 0; i <= max_b_; ++i) off[i] = i * v_;
        ck(cudaMalloc(&d_offsets_, off.size() * sizeof(int)));
        ck(cudaMemcpy(d_offsets_, off.data(), off.size() * sizeof(int),
                      cudaMemcpyHostToDevice));

        // Size temp storage for the largest batch we will ever run.
        size_t bytes = 0;
        cub::DeviceSegmentedReduce::ArgMax(
            nullptr, bytes, (const __half*)nullptr, d_out_,
            max_b_, d_offsets_, d_offsets_ + 1);
        ck(cudaMalloc(&d_temp_, bytes));
        temp_bytes_ = bytes;
    }

    void free() {
        cudaFree(d_out_); cudaFree(d_offsets_); cudaFree(d_temp_);
        d_out_ = nullptr; d_offsets_ = nullptr; d_temp_ = nullptr;
    }

    /// logits: [B, V] fp16 on device. Fills out[0..B) with the argmax token id.
    /// ONE launch, ONE sync, regardless of B.
    void argmax_batched(const __half* d_logits, int B, std::vector<int>& out,
                        cudaStream_t stream) {
        if (B > max_b_) throw std::runtime_error("BatchedPicker: B > max_batch");
        size_t bytes = temp_bytes_;
        cub::DeviceSegmentedReduce::ArgMax(
            d_temp_, bytes, d_logits, d_out_,
            B, d_offsets_, d_offsets_ + 1, stream);

        ck(cudaMemcpyAsync(h_out_.data(), d_out_, (size_t)B * sizeof(Pair),
                           cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));          // the single sync per step

        out.resize(B);
        for (int b = 0; b < B; ++b) out[b] = h_out_[b].key;
    }

private:
    static void ck(cudaError_t e) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("BatchedPicker CUDA: ")
                                     + cudaGetErrorString(e));
    }

    int   v_ = 0, max_b_ = 0;
    Pair* d_out_ = nullptr;
    int*  d_offsets_ = nullptr;
    void* d_temp_ = nullptr;
    size_t temp_bytes_ = 0;
    std::vector<Pair> h_out_;
};
