#include "model.hpp"
#include "kernels.hpp"

#include <cuda_runtime.h>

namespace tllm {

ModelState::ModelState(const Config& cfg) : cfg_(cfg) {
    const size_t D   = cfg.hidden_dim;
    const size_t I   = cfg.intermediate_dim;
    const size_t V   = cfg.vocab_size;
    const size_t L   = cfg.num_layers;
    const size_t Hkv = cfg.num_kv_heads;
    const size_t Dh  = cfg.head_dim;
    const size_t maxS= cfg.max_seq_len;
    const size_t QKV = cfg.qkv_out_dim();
    const size_t Qd  = cfg.q_out_dim();

    x_.resize(D);
    h_.resize(D);
    qkv_.resize(QKV);
    attn_out_.resize(Qd);
    proj_.resize(D);
    gate_up_.resize(2 * I);
    mlp_act_.resize(I);
    logits_.resize(V);

    k_cache_.resize(L * Hkv * maxS * Dh);
    v_cache_.resize(L * Hkv * maxS * Dh);

    next_token_dev_.resize(1);
}

int32_t ModelState::forward(const Weights& w, int32_t token, int32_t pos) {
    using namespace kernels;
    const Config& c = cfg_;
    const int D   = int(c.hidden_dim);
    const int I   = int(c.intermediate_dim);
    const int V   = int(c.vocab_size);
    const int H   = int(c.num_heads);
    const int Hkv = int(c.num_kv_heads);
    const int Dh  = int(c.head_dim);
    const int QKV = int(c.qkv_out_dim());
    const int Qd  = int(c.q_out_dim());
    const int KVd = int(c.kv_out_dim());
    const int maxS= int(c.max_seq_len);
    const size_t layer_cache = size_t(Hkv) * maxS * Dh;

    embed(x_.data(), w.tok_embeddings(), token, D);

    for (uint32_t l = 0; l < c.num_layers; ++l) {
        const LayerWeights& lw = w.layer(l);

        // --- Attention block ---
        rmsnorm(h_.data(), x_.data(), lw.input_norm, D, c.rms_norm_eps);

        // QKV projection (fused). Output layout: [Q | K | V].
        qkv_proj(qkv_.data(), lw.wqkv, lw.wqkv_scales, h_.data(), QKV, D);

        half_t* q_view = qkv_.data();
        half_t* k_view = qkv_.data() + Qd;
        half_t* v_view = qkv_.data() + Qd + KVd;

        rope(q_view, k_view, pos, H, Hkv, Dh, c.rope_theta);

        half_t* k_cache_l = k_cache_.data() + size_t(l) * layer_cache;
        half_t* v_cache_l = v_cache_.data() + size_t(l) * layer_cache;
        kv_append(k_cache_l, v_cache_l, k_view, v_view, pos, Hkv, Dh, maxS);

        attention(attn_out_.data(), q_view, k_cache_l, v_cache_l,
                  pos + 1, H, Hkv, Dh, maxS);

        int8_matmul(proj_.data(), lw.wo, lw.wo_scales, attn_out_.data(), D, Qd);
        residual_add(x_.data(), proj_.data(), D);

        // --- MLP block ---
        rmsnorm(h_.data(), x_.data(), lw.post_attn_norm, D, c.rms_norm_eps);

        int8_matmul(gate_up_.data(), lw.w_gate_up, lw.w_gate_up_scales,
                    h_.data(), 2 * I, D);
        silu_mul(mlp_act_.data(), gate_up_.data(), I);
        int8_matmul(proj_.data(), lw.w_down, lw.w_down_scales, mlp_act_.data(), D, I);
        residual_add(x_.data(), proj_.data(), D);
    }

    rmsnorm(h_.data(), x_.data(), w.final_norm(), D, c.rms_norm_eps);
    int8_matmul(logits_.data(), w.lm_head(), w.lm_head_scales(), h_.data(), V, D);

    argmax(next_token_dev_.data(), logits_.data(), V);

    int32_t next = 0;
    next_token_dev_.copy_to_host(&next, sizeof(int32_t));
    return next;
}

}  // namespace tllm
