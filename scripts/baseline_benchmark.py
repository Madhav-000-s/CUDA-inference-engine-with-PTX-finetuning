"""
HuggingFace FP16 baseline: TinyLlama-1.1B-Chat-v1.0, greedy decode, batch 1.

Reports first-token latency and sustained tokens/sec. This is the number the
CUDA engine must beat by ~2x.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = (
    "The difference between a CPU and a GPU, in a few sentences, is "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "bench" / "baseline.txt")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available. Install PyTorch cu128."
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name(0)}  cc={cap[0]}.{cap[1]}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(dev)
    model.eval()

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)
    prompt_len = input_ids.shape[1]
    print(f"Prompt length: {prompt_len} tokens,  generating {args.max_new_tokens} new tokens")

    @torch.inference_mode()
    def run_once():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1 - t0, out.shape[1] - prompt_len

    @torch.inference_mode()
    def first_token_latency():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for i in range(args.warmup):
        run_once()
        print(f"  warmup {i+1}/{args.warmup} done")

    times = []
    for i in range(args.runs):
        dt, n_new = run_once()
        tps = n_new / dt
        times.append((dt, n_new, tps))
        print(f"  run {i+1}: {dt*1000:.1f} ms for {n_new} tokens -> {tps:.2f} tok/s")

    ftl = first_token_latency()
    print(f"  first-token latency: {ftl*1000:.1f} ms")

    best_tps = max(t[2] for t in times)
    mean_tps = sum(t[2] for t in times) / len(times)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# HF baseline (FP16)\n")
        f.write(f"model: {MODEL_ID}\n")
        f.write(f"device: {torch.cuda.get_device_name(0)} (cc {cap[0]}.{cap[1]})\n")
        f.write(f"prompt_tokens: {prompt_len}\n")
        f.write(f"new_tokens:    {args.max_new_tokens}\n")
        f.write(f"first_token_ms: {ftl*1000:.2f}\n")
        f.write(f"best_tokens_per_sec: {best_tps:.2f}\n")
        f.write(f"mean_tokens_per_sec: {mean_tps:.2f}\n")

    print(f"\nBest: {best_tps:.2f} tok/s   (target for CUDA engine: >= {2*best_tps:.1f} tok/s)")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
