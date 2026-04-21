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
// Hand-tuned inner loop using inline PTX.
//
// Each thread processes 8 contiguous columns per iteration. For in_dim=2048
// and BLOCK=256 (the shape qkv_proj is called with), one iteration covers the
// whole row.
//
// Memory: the naive path emits 8 scalar `ld.global.nc.u8` + 8 scalar
// `ld.global.nc.u16` per thread per iter. We replace that with one 64-bit
// `ld.global.nc.v2.u32` (weights) + one 128-bit `ld.global.nc.v4.u32`
// (activations).
//
// Math: the naive path emits 8 × {cvt.s16.s8, cvt.rn.f32.s16, cvt.f32.f16,
// fma.f32}. We replace with packed ops:
//   - 4 × `cvt.rn.f16x2.s16x2`  (one per s16 pair, vs 8 × scalar cvt)
//   - 4 × `fma.rn.f16x2`        (one per f16 pair, vs 8 × scalar f32 fma)
// The f16x2 accumulator holds at most 4 pair-FMAs per iter; with |s8|<=127
// and typical activation magnitudes ~O(1), the partial sum stays well inside
// FP16 range. Convert to f32 once at the end of the inner loop for the
// cross-warp reduction.
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

    // Two packed f16x2 accumulators (covers 4 columns packed per iter).
    uint32_t acc01 = 0u;  // {col0, col1}
    uint32_t acc23 = 0u;  // {col2, col3}
    uint32_t acc45 = 0u;  // {col4, col5}
    uint32_t acc67 = 0u;  // {col6, col7}

    for (int i0 = tid * 8; i0 < in_dim; i0 += BLOCK * 8) {
        // 8 INT8 weight bytes as 2 × u32 (one 64-bit load).
        uint32_t w_lo, w_hi;
        asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
                     : "=r"(w_lo), "=r"(w_hi)
                     : "l"(W_row + i0));

        // 8 FP16 activations as 4 × u32 (one 128-bit load).
        uint32_t a01, a23, a45, a67;
        asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a01), "=r"(a23), "=r"(a45), "=r"(a67)
                     : "l"(x + i0));

        // Sign-extend each byte of w_lo / w_hi to s32. `cvt.s32.s8` with a b32
        // source is well-defined (reads only the low byte), unlike cvt.s16.s8
        // which ptxas 12.8 rejects when fed from a 32-bit register.
        uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
        asm("cvt.s32.s8 %0, %1;" : "=r"(b0) : "r"(w_lo));
        asm("cvt.s32.s8 %0, %1;" : "=r"(b1) : "r"(w_lo >> 8));
        asm("cvt.s32.s8 %0, %1;" : "=r"(b2) : "r"(w_lo >> 16));
        asm("cvt.s32.s8 %0, %1;" : "=r"(b3) : "r"(w_lo >> 24));
        asm("cvt.s32.s8 %0, %1;" : "=r"(b4) : "r"(w_hi));
        asm("cvt.s32.s8 %0, %1;" : "=r"(b5) : "r"(w_hi >> 8));
        asm("cvt.s32.s8 %0, %1;" : "=r"(b6) : "r"(w_hi >> 16));
        asm("cvt.s32.s8 %0, %1;" : "=r"(b7) : "r"(w_hi >> 24));

        // Convert each s32 byte to f16 (narrowing round-to-nearest). Not the
        // packed cvt.rn.f16x2.s16x2 form — ptxas 12.8 refuses that on sm_120
        // even though the ISA spec says it's legal — but still packed
        // downstream via mov.b32 so the FMA runs in f16x2.
        uint16_t h0, h1, h2, h3, h4, h5, h6, h7;
        asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h0) : "r"(b0));
        asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h1) : "r"(b1));
        asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h2) : "r"(b2));
        asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h3) : "r"(b3));
        asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h4) : "r"(b4));
        asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h5) : "r"(b5));
        asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h6) : "r"(b6));
        asm("cvt.rn.f16.s32 %0, %1;" : "=h"(h7) : "r"(b7));

        // Pack pairs of f16 into f16x2 (b32).
        uint32_t w_f01, w_f23, w_f45, w_f67;
        asm("mov.b32 %0, {%1, %2};" : "=r"(w_f01) : "h"(h0), "h"(h1));
        asm("mov.b32 %0, {%1, %2};" : "=r"(w_f23) : "h"(h2), "h"(h3));
        asm("mov.b32 %0, {%1, %2};" : "=r"(w_f45) : "h"(h4), "h"(h5));
        asm("mov.b32 %0, {%1, %2};" : "=r"(w_f67) : "h"(h6), "h"(h7));

        // Packed FMA: acc += w_f * a  (four fma.rn.f16x2).
        asm("fma.rn.f16x2 %0, %1, %2, %0;"
            : "+r"(acc01) : "r"(w_f01), "r"(a01));
        asm("fma.rn.f16x2 %0, %1, %2, %0;"
            : "+r"(acc23) : "r"(w_f23), "r"(a23));
        asm("fma.rn.f16x2 %0, %1, %2, %0;"
            : "+r"(acc45) : "r"(w_f45), "r"(a45));
        asm("fma.rn.f16x2 %0, %1, %2, %0;"
            : "+r"(acc67) : "r"(w_f67), "r"(a67));
    }

    // Collapse four f16x2 accumulators into a single f32 scalar.
    auto sum_f16x2 = [](uint32_t h2) {
        __half2 v = *reinterpret_cast<__half2*>(&h2);
        return __half2float(__low2half(v)) + __half2float(__high2half(v));
    };
    float acc = sum_f16x2(acc01) + sum_f16x2(acc23)
              + sum_f16x2(acc45) + sum_f16x2(acc67);

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
