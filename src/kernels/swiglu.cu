#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tllm::kernels {

namespace {

// z[i] = silu(gate[i]) * up[i]
//   gate_up is a contiguous buffer of shape [2, I]:
//     gate_up[0 .. I)   = gate
//     gate_up[I .. 2I)  = up
__global__ void silu_mul_kernel(__half* __restrict__ z,
                                const __half* __restrict__ gate_up, int I) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    const float g = __half2float(gate_up[i]);
    const float u = __half2float(gate_up[I + i]);
    const float silu = g / (1.f + __expf(-g));
    z[i] = __float2half(silu * u);
}

}  // namespace

void silu_mul(half_t* z, const half_t* gate_up, int I, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const int grid = (I + BLOCK - 1) / BLOCK;
    silu_mul_kernel<<<grid, BLOCK, 0, stream>>>(z, gate_up, I);
}

}  // namespace tllm::kernels
