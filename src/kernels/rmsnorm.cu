#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tllm::kernels {

namespace {

// One block, BLOCK threads. D must be a multiple of BLOCK/32 per warp-reduce math,
// but the inner loop is strided so any D works.
template <int BLOCK>
__global__ void rmsnorm_kernel(__half* __restrict__ out,
                               const __half* __restrict__ x,
                               const __half* __restrict__ weight,
                               int D, float eps) {
    const int tid = threadIdx.x;

    float sumsq = 0.f;
    for (int i = tid; i < D; i += BLOCK) {
        float v = __half2float(x[i]);
        sumsq += v * v;
    }

    // warp reduce
    for (int mask = 16; mask > 0; mask >>= 1)
        sumsq += __shfl_xor_sync(0xffffffffu, sumsq, mask);

    __shared__ float warp_sums[BLOCK / 32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    if (lane == 0) warp_sums[wid] = sumsq;
    __syncthreads();

    if (wid == 0) {
        float s = (tid < BLOCK / 32) ? warp_sums[lane] : 0.f;
        for (int mask = 16; mask > 0; mask >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, mask);
        if (lane == 0) warp_sums[0] = s;
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / float(D) + eps);
    for (int i = tid; i < D; i += BLOCK) {
        float v = __half2float(x[i]) * rms;
        out[i] = __hmul(__float2half(v), weight[i]);
    }
}

}  // namespace

void rmsnorm(half_t* out, const half_t* x, const half_t* weight,
             int D, float eps, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    rmsnorm_kernel<BLOCK><<<1, BLOCK, 0, stream>>>(out, x, weight, D, eps);
}

}  // namespace tllm::kernels
