#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tllm::kernels {

namespace {

// Generic weight-only INT8 matmul for batch=1.
// One block per output row. Threads in the block reduce across in_dim.
template <int BLOCK>
__global__ void int8_matmul_kernel(__half* __restrict__ out,
                                   const int8_t* __restrict__ W,
                                   const __half* __restrict__ scales,
                                   const __half* __restrict__ x,
                                   int out_dim, int in_dim) {
    const int o   = blockIdx.x;
    const int tid = threadIdx.x;
    if (o >= out_dim) return;

    const int8_t* W_row = W + size_t(o) * in_dim;

    float acc = 0.f;
    for (int i = tid; i < in_dim; i += BLOCK) {
        acc += float(W_row[i]) * __half2float(x[i]);
    }

    for (int mask = 16; mask > 0; mask >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, mask);

    __shared__ float warp_acc[BLOCK / 32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    if (lane == 0) warp_acc[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float a = (tid < BLOCK / 32) ? warp_acc[lane] : 0.f;
        for (int mask = 16; mask > 0; mask >>= 1)
            a += __shfl_xor_sync(0xffffffffu, a, mask);
        if (lane == 0) {
            out[o] = __float2half(a * __half2float(scales[o]));
        }
    }
}

}  // namespace

void int8_matmul(half_t* out, const int8_t* W, const half_t* scales,
                 const half_t* x, int out_dim, int in_dim, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    dim3 grid(out_dim);
    int8_matmul_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(out, W, scales, x, out_dim, in_dim);
}

}  // namespace tllm::kernels
