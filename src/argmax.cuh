/**
 * argmax.cuh  —  GPU argmax over FP16 logits using CUB DeviceReduce
 *
 * Eliminates the 256 KB cudaMemcpy (logits → CPU) + CPU std::max_element
 * that infer_trt.cpp does.  Instead the argmax runs entirely on the GPU
 * and only the winning token ID (4 bytes) comes back to the host.
 *
 * CUB is header-only and ships with every CUDA toolkit ≥ 10.0.
 * No extra CMake linkage needed.
 */

#pragma once

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

// cub::KeyValuePair<int,__half> is the argmax result type
using ArgmaxPair = cub::KeyValuePair<int, __half>;

struct GpuArgmax {
    void*       d_temp    = nullptr;
    size_t      temp_bytes = 0;
    ArgmaxPair* d_result  = nullptr;

    void alloc(int n_elements) {
        // Query how much temp storage CUB needs
        cub::DeviceReduce::ArgMax(nullptr, temp_bytes,
                                  (const __half*)nullptr, (ArgmaxPair*)nullptr,
                                  n_elements);
        cudaMalloc(&d_temp,   temp_bytes);
        cudaMalloc(&d_result, sizeof(ArgmaxPair));
    }

    // Returns the token id with the highest FP16 logit value.
    // d_logits: device pointer to n_elements __half values.
    // stream: CUDA stream to use (synchronises before returning).
    int run(const __half* d_logits, int n_elements, cudaStream_t stream) {
        cub::DeviceReduce::ArgMax(d_temp, temp_bytes,
                                  d_logits, d_result,
                                  n_elements, stream);
        cudaStreamSynchronize(stream);

        ArgmaxPair h_result;
        cudaMemcpy(&h_result, d_result, sizeof(ArgmaxPair), cudaMemcpyDeviceToHost);
        return h_result.key;
    }
};
