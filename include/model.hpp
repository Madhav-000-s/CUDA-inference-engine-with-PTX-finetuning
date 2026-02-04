#pragma once
#include "tensor.hpp"
#include "weights.hpp"

#include <cstdint>

namespace tllm {

// All scratch + KV cache for a single decode stream (batch = 1).
class ModelState {
   public:
    explicit ModelState(const Config& cfg);

    const Config& config() const { return cfg_; }

    // Run one forward step. Updates KV cache at slot `pos`, returns the
    // argmax token id of the logits. Caller tracks `pos`.
    int32_t forward(const Weights& w, int32_t token, int32_t pos);

   private:
    Config cfg_;

    // Working buffers (FP16 unless noted).
    DeviceBuffer<half_t>  x_;            // [D]
    DeviceBuffer<half_t>  h_;            // [D] normed activation
    DeviceBuffer<half_t>  qkv_;          // [(H + 2*Hkv) * Dh]
    DeviceBuffer<half_t>  attn_out_;     // [H*Dh]
    DeviceBuffer<half_t>  proj_;         // [D]  (reused by o_proj and down_proj)
    DeviceBuffer<half_t>  gate_up_;      // [2*I]
    DeviceBuffer<half_t>  mlp_act_;      // [I]
    DeviceBuffer<half_t>  logits_;       // [V]

    // KV caches: [L, Hkv, max_seq, Dh].
    DeviceBuffer<half_t>  k_cache_;
    DeviceBuffer<half_t>  v_cache_;

    // Argmax output.
    DeviceBuffer<int32_t> next_token_dev_;
};

}  // namespace tllm
