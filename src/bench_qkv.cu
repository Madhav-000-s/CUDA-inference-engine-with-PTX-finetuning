// Micro-benchmark for the qkv_proj kernel.
//
// Allocates the TinyLlama QKV shapes on device, fills with random data, and
// times a long loop of kernel invocations with cudaEvents. Used to compare
// the naive and hand-tuned PTX variants without needing a working ncu.
//
// Build: `cmake --build build --target bench_qkv` (naive variant)
//        `cmake --build build_tuned --target bench_qkv` (tuned variant)
// Run  : `./bench_qkv.exe`

#include "kernels.hpp"
#include "tensor.hpp"
#include "config.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using tllm::half_t;

int main() {
    // TinyLlama QKV shape: in_dim = hidden = 2048, out_dim = Q+K+V = 2560.
    const int in_dim  = 2048;
    const int out_dim = 2048 + 256 + 256;  // 2560

    // Host buffers.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dq(-127, 127);
    std::uniform_real_distribution<float> da(-1.f, 1.f);

    std::vector<int8_t>  h_W(size_t(out_dim) * in_dim);
    std::vector<half_t>  h_scales(out_dim);
    std::vector<half_t>  h_x(in_dim);

    for (auto& w : h_W) w = int8_t(dq(rng));
    for (auto& s : h_scales) s = __float2half(0.01f + 0.001f * da(rng));
    for (auto& a : h_x) a = __float2half(da(rng));

    // Device buffers.
    tllm::DeviceBuffer<int8_t> d_W(h_W.size());
    tllm::DeviceBuffer<half_t> d_scales(h_scales.size());
    tllm::DeviceBuffer<half_t> d_x(h_x.size());
    tllm::DeviceBuffer<half_t> d_out(out_dim);

    CUDA_CHECK(cudaMemcpy(d_W.data(),     h_W.data(),     h_W.size()     * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales.data(), h_scales.data(), h_scales.size() * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x.data(),      h_x.data(),      h_x.size()      * sizeof(half_t), cudaMemcpyHostToDevice));

    // Warmup.
    for (int i = 0; i < 100; ++i) {
        tllm::kernels::qkv_proj(d_out.data(), d_W.data(), d_scales.data(),
                                d_x.data(), out_dim, in_dim, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time N invocations.
    const int N = 50000;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < N; ++i) {
        tllm::kernels::qkv_proj(d_out.data(), d_W.data(), d_scales.data(),
                                d_x.data(), out_dim, in_dim, 0);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    const double us_per_call = 1000.0 * ms / N;

    // Roofline sanity: INT8 weights dominate bandwidth for batch-1 decode.
    const double bytes_per_call = double(out_dim) * in_dim;  // int8 weights
    const double gbs = (bytes_per_call / (us_per_call * 1e-6)) / 1e9;

#ifdef TLLM_QKV_HAND_TUNED_PTX
    std::printf("variant: TUNED (inline-PTX)\n");
#else
    std::printf("variant: NAIVE (compiler-generated)\n");
#endif
    std::printf("shape  : out=%d in=%d  (QKV fused)\n", out_dim, in_dim);
    std::printf("calls  : %d (after 100 warmup)\n", N);
    std::printf("total  : %.3f ms\n", ms);
    std::printf("per-call: %.3f us\n", us_per_call);
    std::printf("W bw   : %.2f GB/s  (weight-only, int8)\n", gbs);

    return 0;
}
