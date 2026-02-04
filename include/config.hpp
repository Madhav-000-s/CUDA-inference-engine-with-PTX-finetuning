#pragma once
#include <cstdint>

// TinyLlama-1.1B-Chat-v1.0 architecture constants.
// Matches the model config emitted by scripts/export_weights.py.
namespace tllm {

struct Config {
    uint32_t vocab_size;        // 32000
    uint32_t hidden_dim;        // 2048  (D)
    uint32_t intermediate_dim;  // 5632  (I)
    uint32_t num_layers;        // 22    (L)
    uint32_t num_heads;         // 32    (H)
    uint32_t num_kv_heads;      // 4     (Hkv)
    uint32_t head_dim;          // 64    (Dh)
    uint32_t max_seq_len;       // 2048
    float    rope_theta;        // 10000.0
    float    rms_norm_eps;      // 1e-5

    uint32_t qkv_out_dim() const { return (num_heads + 2 * num_kv_heads) * head_dim; }
    uint32_t q_out_dim()   const { return num_heads * head_dim; }
    uint32_t kv_out_dim()  const { return num_kv_heads * head_dim; }
};

}  // namespace tllm
