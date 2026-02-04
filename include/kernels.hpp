#pragma once
#include "tensor.hpp"

#include <cstdint>

namespace tllm::kernels {

// RMSNorm over a single row of dimension D.
//   out[i] = x[i] * weight[i] * rsqrt( mean(x^2) + eps )
void rmsnorm(half_t* out, const half_t* x, const half_t* weight,
             int D, float eps, cudaStream_t stream = nullptr);

// Weight-only INT8 matmul for a single activation vector (batch = 1).
//   out[o] = sum_i (W_int8[o, i] * scale[o]) * x[i],   shape W = [out_dim, in_dim]
// scales is [out_dim] fp16. Used for o_proj, gate_up, down, lm_head.
void int8_matmul(half_t* out, const int8_t* W, const half_t* scales,
                 const half_t* x, int out_dim, int in_dim,
                 cudaStream_t stream = nullptr);

// Same operation, specialized kernel (separate .cu file) used only for the fused
// QKV projection. Kept isolated so `nvcc -ptx` produces clean output for the
// PTX hand-tuning step.
void qkv_proj(half_t* out, const int8_t* W, const half_t* scales,
              const half_t* x, int out_dim, int in_dim,
              cudaStream_t stream = nullptr);

// Apply RoPE in place on Q [H, Dh] and K [Hkv, Dh] at position `pos`.
void rope(half_t* q, half_t* k, int pos, int H, int Hkv, int Dh,
          float rope_theta, cudaStream_t stream = nullptr);

// Single-query multi-head attention with GQA.
//   out [H, Dh] = softmax( Q [H, Dh] K_cache^T [Hkv, T, Dh] / sqrt(Dh) ) V_cache [Hkv, T, Dh]
// K_cache / V_cache are laid out [Hkv, max_seq, Dh], contents valid for t < T.
void attention(half_t* out, const half_t* q,
               const half_t* k_cache, const half_t* v_cache,
               int T, int H, int Hkv, int Dh, int max_seq,
               cudaStream_t stream = nullptr);

// z[i] = silu(gate_up[i]) * gate_up[I + i]   for i in [0, I).
void silu_mul(half_t* z, const half_t* gate_up, int I, cudaStream_t stream = nullptr);

// out[0] = argmax_i( logits[i] )
void argmax(int32_t* out, const half_t* logits, int V, cudaStream_t stream = nullptr);

// Write one row of the embedding table to x: x[:D] = tok_embeddings[token, :D].
void embed(half_t* x, const half_t* tok_embeddings, int token, int D,
           cudaStream_t stream = nullptr);

// y += x  (both fp16, length D).
void residual_add(half_t* y, const half_t* x, int D, cudaStream_t stream = nullptr);

// Copy current-step K and V into cache slot at position `pos`.
//   K and V are [Hkv, Dh] (contiguous), cache is [Hkv, max_seq, Dh].
void kv_append(half_t* k_cache, half_t* v_cache,
               const half_t* k, const half_t* v,
               int pos, int Hkv, int Dh, int max_seq,
               cudaStream_t stream = nullptr);

}  // namespace tllm::kernels
