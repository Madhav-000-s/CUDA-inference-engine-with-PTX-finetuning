#include "weights.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace tllm {

namespace {
constexpr size_t kHeaderSize = 128;
struct RawHeader {
    char     magic[4];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq_len;
    float    rope_theta;
    float    rms_norm_eps;
};
}  // namespace

Weights::Weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open weights: " + path);
    const std::streamsize file_size = f.tellg();
    f.seekg(0);

    std::vector<uint8_t> host(file_size);
    f.read(reinterpret_cast<char*>(host.data()), file_size);
    if (!f) throw std::runtime_error("failed to read weights file");

    RawHeader hdr{};
    std::memcpy(&hdr, host.data(), sizeof(hdr));
    if (std::memcmp(hdr.magic, "TLLM", 4) != 0) {
        throw std::runtime_error("bad magic in weights file");
    }
    if (hdr.version != 1) {
        throw std::runtime_error("unsupported weights version");
    }
    cfg_ = Config{hdr.vocab_size, hdr.hidden_dim, hdr.intermediate_dim,
                  hdr.num_layers, hdr.num_heads, hdr.num_kv_heads,
                  hdr.head_dim, hdr.max_seq_len, hdr.rope_theta, hdr.rms_norm_eps};

    blob_.resize(file_size);
    blob_.copy_from_host(host.data(), file_size);

    // Build layer pointer table on top of the device blob.
    const uint8_t* base = reinterpret_cast<const uint8_t*>(blob_.data());
    size_t off = kHeaderSize;
    auto eat_fp16 = [&](size_t n) {
        const half_t* p = reinterpret_cast<const half_t*>(base + off);
        off += n * sizeof(half_t);
        return p;
    };
    auto eat_int8 = [&](size_t n) {
        const int8_t* p = reinterpret_cast<const int8_t*>(base + off);
        off += n * sizeof(int8_t);
        return p;
    };

    const uint32_t V  = cfg_.vocab_size;
    const uint32_t D  = cfg_.hidden_dim;
    const uint32_t I  = cfg_.intermediate_dim;
    const uint32_t L  = cfg_.num_layers;
    const uint32_t H  = cfg_.num_heads;
    const uint32_t Hk = cfg_.num_kv_heads;
    const uint32_t Dh = cfg_.head_dim;
    const uint32_t QKV = (H + 2 * Hk) * Dh;

    tok_emb_ = eat_fp16(static_cast<size_t>(V) * D);

    layers_.resize(L);
    for (uint32_t l = 0; l < L; ++l) {
        LayerWeights& lw = layers_[l];
        lw.input_norm       = eat_fp16(D);
        lw.wqkv             = eat_int8(static_cast<size_t>(QKV) * D);
        lw.wqkv_scales      = eat_fp16(QKV);
        lw.wo               = eat_int8(static_cast<size_t>(D) * (H * Dh));
        lw.wo_scales        = eat_fp16(D);
        lw.post_attn_norm   = eat_fp16(D);
        lw.w_gate_up        = eat_int8(static_cast<size_t>(2 * I) * D);
        lw.w_gate_up_scales = eat_fp16(2 * I);
        lw.w_down           = eat_int8(static_cast<size_t>(D) * I);
        lw.w_down_scales    = eat_fp16(D);
    }

    final_norm_      = eat_fp16(D);
    lm_head_         = eat_int8(static_cast<size_t>(V) * D);
    lm_head_scales_  = eat_fp16(V);

    if (off != static_cast<size_t>(file_size)) {
        throw std::runtime_error("weights file size mismatch after parsing");
    }
}

}  // namespace tllm
