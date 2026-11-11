/**
 * layer_kernels.cuh — the non-GEMM pieces of a LLaMA decode step (S2.7).
 *
 * Everything else in a decode layer is cublasHgemm. These four are what cuBLAS
 * cannot do:
 *
 *   rmsnorm     x * rsqrt(mean(x^2) + eps) * w        needs a reduction
 *   rope        rotate (q, k) pairs by position       trig + pairing
 *   silu_mul    silu(gate) * up                       elementwise, fused
 *   embedding   gather one row per token              indexed copy
 *
 * All operate on [B, ...] with one token per sequence (decode, seq = 1).
 *
 * ── fp32 accumulation ────────────────────────────────────────────────────────
 * HuggingFace computes RMSNorm in fp32 and casts back. Summing 2048 squared
 * fp16 values in fp16 both overflows and loses precision, so matching HF
 * requires accumulating in float. The same applies to RoPE's trig.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace layer_kernels {

constexpr int kNormThreads = 256;

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm:  out[b][i] = x[b][i] * rsqrt(mean_i(x[b][i]^2) + eps) * w[i]
//

// ─────────────────────────────────────────────────────────────────────────────
__global__ void rmsnorm(const __half* __restrict__ x,    // [B, hidden]
                        const __half* __restrict__ w,    // [hidden]
                        __half* __restrict__ out,        // [B, hidden]
                        int hidden, float eps) {
    
        const int b = blockIdx.x;
        const int tid = threadIdx.x;
        const int stride = blockDim.x;
        const int n = hidden;
        const int num_blocks = (n + stride - 1) / stride;
        const int start = tid * num_blocks;
        const int end = start + num_blocks;
        const __half* x_row = x + b * hidden;
        __half* out_row = out + b * hidden;
        float sum = 0.0f;
        for (int i = tid; i < n; i+= stride) {
            float val = __half2float(x_row[i]);
            sum += val * val;  
        }

        unsigned int mask = 0xffffffff;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        __shared__ float shared_sum[32];
        __shared__ float shared_scale;
        int lane_id = tid % warpSize;
        int warp_id = tid / warpSize;
        if (lane_id == 0) {
            shared_sum[warp_id] = sum;
        }
        __syncthreads();
        if (warp_id == 0) {
            sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(mask, sum, offset);
            }
            if (lane_id == 0) {    
                shared_scale = rsqrtf(sum / n + eps);
            }
        }
        __syncthreads();
        float inv_rms = shared_scale;
        for (int i = tid; i < n; i += stride) {
            float val = __half2float(x_row[i]);
            float w_val = __half2float(w[i]);
            out_row[i] = __float2half(__half2float(x_row[i]) * inv_rms * __half2float(w[i]));
        }


}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE (rotary position embedding), LLaMA/NeoX pairing.
//
// A head's D values are split in HALF; element i is paired with element i+D/2:
//
//   freq  = 1 / theta^(2i/D)          i in [0, D/2)
//   angle = pos * freq
//   out[i]     = x[i]     * cos(angle) - x[i+D/2] * sin(angle)
//   out[i+D/2] = x[i+D/2] * cos(angle) + x[i]     * sin(angle)
//
// NOTE the pairing: NOT (0,1), (2,3)... but (0, D/2), (1, D/2+1), ...
// Getting this wrong produces fluent-but-wrong output, the usual failure mode.
//
// Applied to q [B, Hq, D] and k [B, Hkv, D] IN PLACE, before k is written to
// the cache — the cache stores post-RoPE keys.
//

// ─────────────────────────────────────────────────────────────────────────────
__global__ void rope_inplace(__half* __restrict__ x,     // [B, H, D]
                             const int* __restrict__ positions,  // [B]
                             int H, int D, float theta) {
    const int bh = blockIdx.x;              // B*H blocks
    const int b  = bh / H, h = bh % H;
    const int pos = positions[b];
    __half* row = x + ((size_t)b * H + h) * D;

    for (int i = threadIdx.x; i < D / 2; i += blockDim.x) {
        const float freq  = 1.0f / powf(theta, (2.0f * i) / D);
        float s, c;
        __sincosf(pos * freq, &s, &c);
        const float xi = __half2float(row[i]);
        const float xj = __half2float(row[i + D / 2]);
        row[i]         = __float2half(xi * c - xj * s);
        row[i + D / 2] = __float2half(xj * c + xi * s);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SiLU-and-multiply:  out[i] = silu(gate[i]) * up[i],  silu(v) = v * sigmoid(v)
// Fusing avoids a second pass over 8192 elements per token.
//

// ─────────────────────────────────────────────────────────────────────────────
__global__ void silu_mul(const __half* __restrict__ gate,  // [B, inter]
                         const __half* __restrict__ up,    // [B, inter]
                         __half* __restrict__ out,         // [B, inter]
                         int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float s = g / (1.0f + expf(-g));  // silu
        out[i] = __float2half(s * u);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding lookup: out[b] = table[ token_ids[b] ]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void embedding(const __half* __restrict__ table,  // [vocab, hidden]
                          const int* __restrict__ token_ids, // [B]
                          __half* __restrict__ out,          // [B, hidden]
                          int hidden, int n) {               // n = B * hidden
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        const int b = idx / hidden;
        const int i = idx % hidden;
        int token_id = token_ids[b];
        out[idx] = table[(size_t)token_id * hidden + i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Residual add: x[i] += y[i]   (in place on x)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void residual_add(__half* __restrict__ x,
                             const __half* __restrict__ y, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        x[tid] = __float2half(__half2float(x[tid]) +  __half2float(y[tid]));
    }
}

}  // namespace layer_kernels

// ── Launchers ────────────────────────────────────────────────────────────────

inline void launch_rmsnorm(const __half* x, const __half* w, __half* out,
                           int B, int hidden, float eps, cudaStream_t s) {
    
    layer_kernels::rmsnorm<<<B, layer_kernels::kNormThreads, 0, s>>>(x, w, out, hidden, eps);
}

inline void launch_rope(__half* x, const int* positions,
                        int B, int H, int D, float theta, cudaStream_t s) {
    
    layer_kernels::rope_inplace<<<B * H, 256, 0, s>>>(x, positions, H, D, theta);
}

inline void launch_silu_mul(const __half* gate, const __half* up, __half* out,
                            int B, int inter, cudaStream_t s) {
    layer_kernels::silu_mul<<<(B * inter + 255) / 256, 256, 0, s>>>(gate, up, out, B * inter);
}

inline void launch_embedding(const __half* table, const int* ids, __half* out,
                             int B, int hidden, cudaStream_t s) {
    const int n = B * hidden;
    layer_kernels::embedding<<<(n + 255) / 256, 256, 0, s>>>(table, ids, out, hidden, n);
}


inline void launch_residual_add(__half* x, const __half* y, int n,
                                cudaStream_t s) {
    layer_kernels::residual_add<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
}
