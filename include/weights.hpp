#pragma once
#include "config.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tllm {

// Per-layer device pointers into Weights::blob_.
struct LayerWeights {
    const half_t*  input_norm;           // [D]
    const int8_t*  wqkv;                 // [(H + 2*Hkv) * Dh, D]
    const half_t*  wqkv_scales;          // [(H + 2*Hkv) * Dh]
    const int8_t*  wo;                   // [D, H*Dh]
    const half_t*  wo_scales;            // [D]
    const half_t*  post_attn_norm;       // [D]
    const int8_t*  w_gate_up;            // [2*I, D]
    const half_t*  w_gate_up_scales;     // [2*I]
    const int8_t*  w_down;               // [D, I]
    const half_t*  w_down_scales;        // [D]
};

class Weights {
   public:
    // Loads the binary file produced by scripts/export_weights.py onto the GPU.
    explicit Weights(const std::string& path);

    const Config&              config() const { return cfg_; }
    const half_t*              tok_embeddings() const { return tok_emb_; }
    const LayerWeights&        layer(uint32_t i) const { return layers_[i]; }
    const half_t*              final_norm() const { return final_norm_; }
    const int8_t*              lm_head() const { return lm_head_; }
    const half_t*              lm_head_scales() const { return lm_head_scales_; }

    // Full device buffer (owns all weight memory). Kept private via accessor only for debug.
    const DeviceBuffer<uint8_t>& blob() const { return blob_; }

   private:
    Config                      cfg_{};
    DeviceBuffer<uint8_t>       blob_;
    std::vector<LayerWeights>   layers_;
    const half_t*               tok_emb_         = nullptr;
    const half_t*               final_norm_      = nullptr;
    const int8_t*               lm_head_         = nullptr;
    const half_t*               lm_head_scales_  = nullptr;
};

}  // namespace tllm
