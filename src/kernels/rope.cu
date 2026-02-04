#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace tllm::kernels {

namespace {

// One block per "vector" (H Q-heads + Hkv K-heads = H + Hkv blocks).
// Each thread handles one pair (d, d + Dh/2) for d in [0, Dh/2).
// Rotation (HF Llama convention):
//   x'[d]        = x[d] * c - x[d + Dh/2] * s
//   x'[d + Dh/2] = x[d + Dh/2] * c + x[d] * s
// where angle = pos / rope_theta^(2d/Dh), c = cos(angle), s = sin(angle).
__global__ void rope_kernel(__half* __restrict__ q, __half* __restrict__ k,
                            int pos, int H, int Hkv, int Dh, float rope_theta) {
    const int head = blockIdx.x;
    const int d    = threadIdx.x;
    if (d >= Dh / 2) return;

    __half* vec;
    if (head < H) {
        vec = q + head * Dh;
    } else {
        vec = k + (head - H) * Dh;
    }

    const float inv_freq = __powf(rope_theta, -2.f * float(d) / float(Dh));
    const float angle    = float(pos) * inv_freq;
    const float c = __cosf(angle);
    const float s = __sinf(angle);

    const float x0 = __half2float(vec[d]);
    const float x1 = __half2float(vec[d + Dh / 2]);
    vec[d]          = __float2half(x0 * c - x1 * s);
    vec[d + Dh / 2] = __float2half(x1 * c + x0 * s);
}

}  // namespace

void rope(half_t* q, half_t* k, int pos, int H, int Hkv, int Dh,
          float rope_theta, cudaStream_t stream) {
    dim3 grid(H + Hkv);
    dim3 block(Dh / 2);
    rope_kernel<<<grid, block, 0, stream>>>(q, k, pos, H, Hkv, Dh, rope_theta);
}

}  // namespace tllm::kernels
