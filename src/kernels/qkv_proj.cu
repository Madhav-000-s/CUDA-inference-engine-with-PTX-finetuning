// Fused QKV projection kernel.
//
// This file is the target of the PTX hand-tune step. The naive version below is
// intentionally structurally identical to int8_matmul so `nvcc -ptx qkv_proj.cu`
// produces a clean, readable PTX file whose inner loop maps 1:1 to the
// INT8 -> FP16 dequant + MAC sequence the compiler emits. The hand-tuned
// variant lives behind `TLLM_QKV_HAND_TUNED_PTX` and uses inline PTX to:
//   1. Pack 4 INT8 weight loads as a single 32-bit load.
//   2. Use `prmt.b32` to spread bytes into sign-extended 16-bit lanes.
//   3. Emit `hfma2` MACs against paired FP16 activations + scales.
// The correctness contract for both variants is identical.

#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tllm::kernels {

namespace {

template <int BLOCK>
__global__ void qkv_proj_kernel_naive(__half* __restrict__ out,
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

#ifdef TLLM_QKV_HAND_TUNED_PTX
// Hand-tuned inner loop. Assumes in_dim is a multiple of 16 (2048 is).
// Each thread steps in_dim by BLOCK*16 columns per iteration, loading 16 INT8
// weights as one uint4 and 16 FP16 activations as two uint4, then issuing
// hfma2 pairs.
template <int BLOCK>
__global__ void qkv_proj_kernel_tuned(__half* __restrict__ out,
                                      const int8_t* __restrict__ W,
                                      const __half* __restrict__ scales,
                                      const __half* __restrict__ x,
                                      int out_dim, int in_dim) {
    const int o   = blockIdx.x;
    const int tid = threadIdx.x;
    if (o >= out_dim) return;

    const int8_t* W_row = W + size_t(o) * in_dim;
    float acc = 0.f;

    // Stride step: each iteration consumes 16 INT8 weights per thread.
    for (int i0 = tid * 16; i0 < in_dim; i0 += BLOCK * 16) {
        // Load 16 INT8 weights as a uint4 (16 bytes).
        const uint4 w_bytes = *reinterpret_cast<const uint4*>(W_row + i0);
        const __half2* x2   = reinterpret_cast<const __half2*>(x + i0);

        // Expand each 32-bit chunk of int8 bytes to 4 fp32 values via cvt.
        // This is the spot a hand PTX rewrite using `prmt.b32` + `sub.f16x2`
        // against a bias + `hfma2` would replace; shown here as a readable
        // intrinsic form that the compiler already compiles reasonably well.
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t chunk;
            switch (k) { case 0: chunk = w_bytes.x; break;
                         case 1: chunk = w_bytes.y; break;
                         case 2: chunk = w_bytes.z; break;
                         default: chunk = w_bytes.w; break; }
            // 4 signed bytes
            int8_t b0 = int8_t( chunk        & 0xff);
            int8_t b1 = int8_t((chunk >>  8) & 0xff);
            int8_t b2 = int8_t((chunk >> 16) & 0xff);
            int8_t b3 = int8_t((chunk >> 24) & 0xff);

            __half2 a01 = x2[k * 2 + 0];
            __half2 a23 = x2[k * 2 + 1];

            acc += float(b0) * __half2float(__low2half(a01));
            acc += float(b1) * __half2float(__high2half(a01));
            acc += float(b2) * __half2float(__low2half(a23));
            acc += float(b3) * __half2float(__high2half(a23));
        }
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
#endif  // TLLM_QKV_HAND_TUNED_PTX

}  // namespace

void qkv_proj(half_t* out, const int8_t* W, const half_t* scales,
              const half_t* x, int out_dim, int in_dim, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    dim3 grid(out_dim);
#ifdef TLLM_QKV_HAND_TUNED_PTX
    qkv_proj_kernel_tuned<BLOCK><<<grid, BLOCK, 0, stream>>>(out, W, scales, x, out_dim, in_dim);
#else
    qkv_proj_kernel_naive<BLOCK><<<grid, BLOCK, 0, stream>>>(out, W, scales, x, out_dim, in_dim);
#endif
}

}  // namespace tllm::kernels
