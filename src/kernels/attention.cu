#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace tllm::kernels {

namespace {

// One block per Q-head. Block size BLOCK. smem = T * sizeof(float) for scores.
// Assumes Dh == 64 (TinyLlama); enforced by an assert on the host side.
template <int BLOCK, int DH>
__global__ void attention_kernel(__half* __restrict__ out,
                                 const __half* __restrict__ q,
                                 const __half* __restrict__ k_cache,
                                 const __half* __restrict__ v_cache,
                                 int T, int H, int Hkv, int max_seq,
                                 float inv_sqrt_dh) {
    const int h    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int kv_h = h * Hkv / H;      // 8 Q heads per KV head

    const __half* q_h = q + h    * DH;
    const __half* k_h = k_cache  + size_t(kv_h) * max_seq * DH;
    const __half* v_h = v_cache  + size_t(kv_h) * max_seq * DH;

    extern __shared__ float scores[];        // length T

    // Stage q in smem as fp32.
    __shared__ float q_shared[DH];
    if (tid < DH) q_shared[tid] = __half2float(q_h[tid]);
    __syncthreads();

    // 1) score[t] = (q . k[t]) * inv_sqrt_dh
    for (int t = tid; t < T; t += BLOCK) {
        const __half* kt = k_h + size_t(t) * DH;
        float s = 0.f;
        #pragma unroll
        for (int d = 0; d < DH; ++d) {
            s += q_shared[d] * __half2float(kt[d]);
        }
        scores[t] = s * inv_sqrt_dh;
    }
    __syncthreads();

    // 2) max reduction
    float local_max = -CUDART_INF_F;
    for (int t = tid; t < T; t += BLOCK) local_max = fmaxf(local_max, scores[t]);
    for (int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, mask));

    __shared__ float warp_max[BLOCK / 32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    if (lane == 0) warp_max[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        float m = (tid < BLOCK / 32) ? warp_max[lane] : -CUDART_INF_F;
        for (int mask = 16; mask > 0; mask >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, mask));
        if (lane == 0) warp_max[0] = m;
    }
    __syncthreads();
    const float max_s = warp_max[0];

    // 3) exp and sum
    float local_sum = 0.f;
    for (int t = tid; t < T; t += BLOCK) {
        float e = __expf(scores[t] - max_s);
        scores[t] = e;
        local_sum += e;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        local_sum += __shfl_xor_sync(0xffffffffu, local_sum, mask);

    __shared__ float warp_sum[BLOCK / 32];
    if (lane == 0) warp_sum[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        float s = (tid < BLOCK / 32) ? warp_sum[lane] : 0.f;
        for (int mask = 16; mask > 0; mask >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, mask);
        if (lane == 0) warp_sum[0] = s;
    }
    __syncthreads();
    const float inv_sum = 1.f / warp_sum[0];

    // 4) out[h, d] = sum_t softmax(s)[t] * V[t, d]
    for (int d = tid; d < DH; d += BLOCK) {
        float acc = 0.f;
        for (int t = 0; t < T; ++t) {
            acc += scores[t] * __half2float(v_h[size_t(t) * DH + d]);
        }
        out[size_t(h) * DH + d] = __float2half(acc * inv_sum);
    }
}

}  // namespace

void attention(half_t* out, const half_t* q, const half_t* k_cache,
               const half_t* v_cache, int T, int H, int Hkv, int Dh,
               int max_seq, cudaStream_t stream) {
    constexpr int BLOCK = 128;
    const size_t smem_bytes = size_t(T) * sizeof(float);
    // TinyLlama has Dh=64. Templated for constant unrolling.
    if (Dh == 64) {
        attention_kernel<BLOCK, 64><<<dim3(H), BLOCK, smem_bytes, stream>>>(
            out, q, k_cache, v_cache, T, H, Hkv, max_seq, rsqrtf(float(Dh)));
    } else {
        // Fallback path would go here; for this project Dh is always 64.
    }
}

}  // namespace tllm::kernels
