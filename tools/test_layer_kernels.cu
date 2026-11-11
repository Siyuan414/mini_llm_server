/**
 * test_layer_kernels.cu — S2.7 unit tests for the non-GEMM kernels.
 *
 * Each kernel is checked against a CPU reference computed in double precision,
 * so a failure means the kernel is wrong, not that fp16 rounded.
 *
 * Build:
 *   nvcc -std=c++17 -I kernels tools/test_layer_kernels.cu -o build/test_layers \
 *        -arch=sm_120
 * Run:
 *   ./build/test_layers
 */

#include "layer_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                   \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    exit(1);} } while(0)

static int failures = 0;

static void report(const char* name, double worst, double tol) {
    const bool ok = worst <= tol;
    printf("  %-14s max|diff| = %.6f   %s\n", name, worst, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

template <typename T>
static T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static std::vector<float> download(const __half* d, size_t n) {
    std::vector<__half> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = __half2float(h[i]);
    return f;
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.5f);
    auto rnd_half = [&](size_t n) {
        std::vector<__half> v(n);
        for (auto& x : v) x = __float2half(nd(rng));
        return v;
    };

    // ── RMSNorm ─────────────────────────────────────────────────────────────
    {
        printf("rmsnorm\n");
        const int B = 3, hidden = 2048;
        const float eps = 1e-5f;
        auto hx = rnd_half((size_t)B * hidden);
        auto hw = rnd_half(hidden);

        __half* dx = upload(hx);
        __half* dw = upload(hw);
        __half* dout = nullptr;
        CK(cudaMalloc(&dout, (size_t)B * hidden * sizeof(__half)));

        launch_rmsnorm(dx, dw, dout, B, hidden, eps, 0);
        CK(cudaDeviceSynchronize());
        auto got = download(dout, (size_t)B * hidden);

        double worst = 0;
        for (int b = 0; b < B; ++b) {
            double ss = 0;
            for (int i = 0; i < hidden; ++i) {
                const double v = __half2float(hx[(size_t)b * hidden + i]);
                ss += v * v;
            }
            const double scale = 1.0 / std::sqrt(ss / hidden + eps);
            for (int i = 0; i < hidden; ++i) {
                const double want = __half2float(hx[(size_t)b * hidden + i])
                                  * scale * __half2float(hw[i]);
                worst = std::max(worst,
                                 std::fabs(want - got[(size_t)b * hidden + i]));
            }
        }
        report("rmsnorm", worst, 2e-2);
        cudaFree(dx); cudaFree(dw); cudaFree(dout);
    }

    // ── RoPE ────────────────────────────────────────────────────────────────
    {
        printf("rope\n");
        const int B = 2, H = 4, D = 64;
        const float theta = 500000.f;             // LLaMA-3.2 rope_theta
        auto hx = rnd_half((size_t)B * H * D);
        std::vector<int> pos{7, 130};

        __half* dx = upload(hx);
        int* dpos  = upload(pos);
        launch_rope(dx, dpos, B, H, D, theta, 0);
        CK(cudaDeviceSynchronize());
        auto got = download(dx, (size_t)B * H * D);

        double worst = 0;
        for (int b = 0; b < B; ++b)
          for (int h = 0; h < H; ++h)
            for (int i = 0; i < D / 2; ++i) {
                const double freq  = 1.0 / std::pow((double)theta,
                                                    (2.0 * i) / D);
                const double angle = pos[b] * freq;
                const double c = std::cos(angle), s = std::sin(angle);
                const size_t base = ((size_t)b * H + h) * D;
                const double a = __half2float(hx[base + i]);
                const double d = __half2float(hx[base + i + D / 2]);
                worst = std::max(worst, std::fabs((a * c - d * s) - got[base + i]));
                worst = std::max(worst,
                                 std::fabs((d * c + a * s) - got[base + i + D/2]));
            }
        report("rope", worst, 2e-2);
        cudaFree(dx); cudaFree(dpos);
    }

    // ── SiLU-mul ────────────────────────────────────────────────────────────
    {
        printf("silu_mul\n");
        const int B = 3, inter = 8192;
        const size_t n = (size_t)B * inter;
        auto hg = rnd_half(n), hu = rnd_half(n);

        __half* dg = upload(hg);
        __half* du = upload(hu);
        __half* dout = nullptr;
        CK(cudaMalloc(&dout, n * sizeof(__half)));
        launch_silu_mul(dg, du, dout, B, inter, 0);
        CK(cudaDeviceSynchronize());
        auto got = download(dout, n);

        double worst = 0;
        for (size_t i = 0; i < n; ++i) {
            const double g = __half2float(hg[i]), u = __half2float(hu[i]);
            const double want = (g / (1.0 + std::exp(-g))) * u;
            worst = std::max(worst, std::fabs(want - got[i]));
        }
        report("silu_mul", worst, 2e-2);
        cudaFree(dg); cudaFree(du); cudaFree(dout);
    }

    // ── Embedding ───────────────────────────────────────────────────────────
    {
        printf("embedding\n");
        const int B = 4, hidden = 2048, vocab = 1000;
        auto table = rnd_half((size_t)vocab * hidden);
        std::vector<int> ids{0, 999, 42, 7};

        __half* dt  = upload(table);
        int*    did = upload(ids);
        __half* dout = nullptr;
        CK(cudaMalloc(&dout, (size_t)B * hidden * sizeof(__half)));
        launch_embedding(dt, did, dout, B, hidden, 0);
        CK(cudaDeviceSynchronize());
        auto got = download(dout, (size_t)B * hidden);

        double worst = 0;
        for (int b = 0; b < B; ++b)
            for (int i = 0; i < hidden; ++i) {
                const double want =
                    __half2float(table[(size_t)ids[b] * hidden + i]);
                worst = std::max(worst,
                                 std::fabs(want - got[(size_t)b * hidden + i]));
            }
        report("embedding", worst, 0.0);      // an exact copy: must be exact
        cudaFree(dt); cudaFree(did); cudaFree(dout);
    }

    // ── Residual add ────────────────────────────────────────────────────────
    {
        printf("residual_add\n");
        const size_t n = 4096;
        auto hx = rnd_half(n), hy = rnd_half(n);
        __half* dx = upload(hx);
        __half* dy = upload(hy);
        launch_residual_add(dx, dy, (int)n, 0);
        CK(cudaDeviceSynchronize());
        auto got = download(dx, n);

        double worst = 0;
        for (size_t i = 0; i < n; ++i) {
            const double want = (double)__half2float(hx[i])
                              + (double)__half2float(hy[i]);
            worst = std::max(worst, std::fabs(want - got[i]));
        }
        report("residual_add", worst, 2e-2);
        cudaFree(dx); cudaFree(dy);
    }

    printf("\n%s\n", failures ? "FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
