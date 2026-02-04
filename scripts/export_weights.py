"""
Export TinyLlama-1.1B-Chat-v1.0 weights to a flat binary for the C++ engine.

Quantization: weight-only INT8, symmetric, per-output-channel.
  For each linear W [out, in]:
    scale[o]     = max(|W[o, :]|) / 127
    W_int8[o, i] = round(W[o, i] / scale[o])  clipped to [-127, 127]
  Dequant at inference: W_fp16[o, i] = W_int8[o, i] * scale[o]

Binary layout (little-endian):
  Header (128 bytes, padded with zeros):
    char[4]   magic          = "TLLM"
    uint32    version        = 1
    uint32    vocab_size
    uint32    hidden_dim
    uint32    intermediate_dim
    uint32    num_layers
    uint32    num_heads
    uint32    num_kv_heads
    uint32    head_dim
    uint32    max_seq_len
    float32   rope_theta
    float32   rms_norm_eps

  Tensors (concatenated, in the order emitted below):
    tok_embeddings              fp16  [V, D]
    for l in 0..L:
      input_layernorm           fp16  [D]
      wqkv                      int8  [(H + 2*Hkv) * Dh, D]
      wqkv_scales               fp16  [(H + 2*Hkv) * Dh]
      wo                        int8  [D, H*Dh]
      wo_scales                 fp16  [D]
      post_attention_layernorm  fp16  [D]
      w_gate_up                 int8  [2*I, D]        (gate and up projections stacked on output dim)
      w_gate_up_scales          fp16  [2*I]
      w_down                    int8  [D, I]
      w_down_scales             fp16  [D]
    final_norm                  fp16  [D]
    lm_head                     int8  [V, D]
    lm_head_scales              fp16  [V]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_DIR = Path(__file__).resolve().parent.parent / "weights"
HEADER_SIZE = 128


def quantize_per_channel(W: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """W is [out, in] float tensor. Returns (int8 weights, fp16 scales)."""
    W = W.detach().to(torch.float32).cpu()
    absmax = W.abs().amax(dim=1).clamp(min=1e-8)           # [out]
    scale = absmax / 127.0                                 # [out]
    q = torch.round(W / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return q.numpy(), scale.to(torch.float16).numpy()


def write_fp16(f, t: torch.Tensor):
    arr = t.detach().to(torch.float16).cpu().numpy()
    f.write(arr.tobytes())


def write_int8(f, arr: np.ndarray):
    assert arr.dtype == np.int8
    f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR / "tinyllama-int8.bin")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16",
                    help="Load model in this dtype before quantizing.")
    ap.add_argument("--verify", action="store_true",
                    help="After export, dequantize one layer and report cosine similarity vs original.")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok_dir = Path(__file__).resolve().parent.parent / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.save_pretrained(tok_dir)
    model.eval()
    cfg = model.config

    V = cfg.vocab_size
    D = cfg.hidden_size
    I = cfg.intermediate_size
    L = cfg.num_hidden_layers
    H = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    Dh = D // H
    max_seq = cfg.max_position_embeddings
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
    rms_eps = float(cfg.rms_norm_eps)

    print(f"  V={V} D={D} I={I} L={L} H={H} Hkv={Hkv} Dh={Dh} max_seq={max_seq}")
    assert Hkv * Dh * 2 + H * Dh == (H + 2 * Hkv) * Dh

    sd = model.state_dict()
    first_wqkv_ref = None  # for optional verification

    with open(args.out, "wb") as f:
        header = struct.pack(
            "<4sIIIIIIIIIff",
            b"TLLM", 1, V, D, I, L, H, Hkv, Dh, max_seq, rope_theta, rms_eps,
        )
        assert len(header) <= HEADER_SIZE
        f.write(header + b"\x00" * (HEADER_SIZE - len(header)))

        write_fp16(f, sd["model.embed_tokens.weight"])

        for l in range(L):
            p = f"model.layers.{l}."
            write_fp16(f, sd[p + "input_layernorm.weight"])

            Wq = sd[p + "self_attn.q_proj.weight"]
            Wk = sd[p + "self_attn.k_proj.weight"]
            Wv = sd[p + "self_attn.v_proj.weight"]
            Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
            qkv_int8, qkv_scales = quantize_per_channel(Wqkv)
            if l == 0:
                first_wqkv_ref = (Wqkv, qkv_int8, qkv_scales)
            write_int8(f, qkv_int8)
            f.write(qkv_scales.tobytes())

            Wo = sd[p + "self_attn.o_proj.weight"]
            wo_int8, wo_scales = quantize_per_channel(Wo)
            write_int8(f, wo_int8)
            f.write(wo_scales.tobytes())

            write_fp16(f, sd[p + "post_attention_layernorm.weight"])

            Wg = sd[p + "mlp.gate_proj.weight"]
            Wu = sd[p + "mlp.up_proj.weight"]
            Wgu = torch.cat([Wg, Wu], dim=0)
            gu_int8, gu_scales = quantize_per_channel(Wgu)
            write_int8(f, gu_int8)
            f.write(gu_scales.tobytes())

            Wd = sd[p + "mlp.down_proj.weight"]
            wd_int8, wd_scales = quantize_per_channel(Wd)
            write_int8(f, wd_int8)
            f.write(wd_scales.tobytes())

            if (l + 1) % 5 == 0 or l == L - 1:
                print(f"  layer {l+1}/{L} written")

        write_fp16(f, sd["model.norm.weight"])

        Wlm = sd["lm_head.weight"]
        lm_int8, lm_scales = quantize_per_channel(Wlm)
        write_int8(f, lm_int8)
        f.write(lm_scales.tobytes())

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.out} ({size_mb:.1f} MiB)")

    if args.verify and first_wqkv_ref is not None:
        Wqkv, qkv_int8, qkv_scales = first_wqkv_ref
        deq = qkv_int8.astype(np.float32) * qkv_scales.astype(np.float32)[:, None]
        ref = Wqkv.to(torch.float32).cpu().numpy()
        cos = (deq * ref).sum() / (np.linalg.norm(deq) * np.linalg.norm(ref) + 1e-12)
        max_abs = np.abs(deq - ref).max()
        print(f"Verify layer 0 QKV: cosine={cos:.6f}  max_abs_err={max_abs:.4f}")


if __name__ == "__main__":
    main()
