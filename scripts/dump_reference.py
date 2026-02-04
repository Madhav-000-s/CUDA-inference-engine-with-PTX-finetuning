"""
Run HuggingFace TinyLlama on a fixed short prompt and dump intermediate tensors
(first layer's input_norm output, qkv, attention output, mlp output, and the
final logits) to .npy files. Used by a later correctness harness to diff each
CUDA kernel against the PyTorch reference.

Usage:
    python scripts/dump_reference.py --out bench/ref
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Hello"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "bench" / "ref")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(dev)
    model.eval()

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(dev)
    np.save(args.out / "prompt_ids.npy", ids.cpu().numpy())

    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().float().cpu()
        return fn

    layer0 = model.model.layers[0]
    layer0.input_layernorm.register_forward_hook(hook("layer0.input_norm_out"))
    layer0.self_attn.register_forward_hook(hook("layer0.attn_out"))
    layer0.post_attention_layernorm.register_forward_hook(hook("layer0.post_norm_out"))
    layer0.mlp.register_forward_hook(hook("layer0.mlp_out"))
    model.model.norm.register_forward_hook(hook("final_norm_out"))

    with torch.inference_mode():
        out = model(ids, use_cache=True)

    captured["logits"] = out.logits[:, -1, :].detach().float().cpu()

    for name, t in captured.items():
        np.save(args.out / f"{name}.npy", t.numpy())
        print(f"saved {name}  shape={tuple(t.shape)}")

    # Also dump top-5 token ids for the next token so the C++ engine can
    # assert parity on the argmax.
    top5 = out.logits[0, -1].topk(5).indices.cpu().numpy()
    np.save(args.out / "top5_next_token.npy", top5)
    print(f"top-5 next tokens: {top5}")


if __name__ == "__main__":
    main()
