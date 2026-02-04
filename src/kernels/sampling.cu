#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace tllm::kernels {

namespace {

template <int BLOCK>
__global__ void argmax_kernel(int32_t* __restrict__ out,
                              const __half* __restrict__ logits, int V) {
    __shared__ float s_val[BLOCK];
    __shared__ int   s_idx[BLOCK];
    const int tid = threadIdx.x;

    float best_v = -CUDART_INF_F;
    int   best_i = 0;
    for (int i = tid; i < V; i += BLOCK) {
        float v = __half2float(logits[i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int step = BLOCK / 2; step > 0; step >>= 1) {
        if (tid < step) {
            float ov = s_val[tid + step];
            if (ov > s_val[tid] || (ov == s_val[tid] && s_idx[tid + step] < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = s_idx[tid + step];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out = s_idx[0];
}

__global__ void embed_kernel(__half* __restrict__ x,
                             const __half* __restrict__ tok_emb,
                             int token, int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) x[i] = tok_emb[size_t(token) * D + i];
}

__global__ void residual_add_kernel(__half* __restrict__ y,
                                    const __half* __restrict__ x, int D) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) y[i] = __hadd(y[i], x[i]);
}

__global__ void kv_append_kernel(__half* __restrict__ k_cache,
                                 __half* __restrict__ v_cache,
                                 const __half* __restrict__ k,
                                 const __half* __restrict__ v,
                                 int pos, int Hkv, int Dh, int max_seq) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Hkv * Dh;
    if (idx >= total) return;
    const int h = idx / Dh;
    const int d = idx % Dh;
    const size_t cache_off = size_t(h) * max_seq * Dh + size_t(pos) * Dh + d;
    k_cache[cache_off] = k[idx];
    v_cache[cache_off] = v[idx];
}

}  // namespace

void argmax(int32_t* out, const half_t* logits, int V, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    argmax_kernel<BLOCK><<<1, BLOCK, 0, stream>>>(out, logits, V);
}

void embed(half_t* x, const half_t* tok_emb, int token, int D, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    embed_kernel<<<(D + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(x, tok_emb, token, D);
}

void residual_add(half_t* y, const half_t* x, int D, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    residual_add_kernel<<<(D + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(y, x, D);
}

void kv_append(half_t* k_cache, half_t* v_cache, const half_t* k, const half_t* v,
               int pos, int Hkv, int Dh, int max_seq, cudaStream_t stream) {
    constexpr int BLOCK = 128;
    const int total = Hkv * Dh;
    kv_append_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        k_cache, v_cache, k, v, pos, Hkv, Dh, max_seq);
}

}  // namespace tllm::kernels
