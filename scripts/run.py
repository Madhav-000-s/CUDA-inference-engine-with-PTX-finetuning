"""
Wrapper that tokenizes a prompt, invokes the C++ inference binary, and
decodes the generated token IDs back to text.

Usage:
    python scripts/run.py "The difference between a CPU and a GPU is" --max-tokens 128
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "build" / "Release" / "inference.exe"
DEFAULT_WEIGHTS = ROOT / "weights" / "tinyllama-int8.bin"
TOK_DIR = ROOT / "tokenizer"


def find_binary(explicit: Path | None) -> Path:
    if explicit and explicit.exists():
        return explicit
    candidates = [
        ROOT / "build" / "Release" / "inference.exe",
        ROOT / "build" / "inference.exe",
        ROOT / "build" / "Release" / "inference",
        ROOT / "build" / "inference",
    ]
    for p in candidates:
        if p.exists():
            return p
    sys.exit(f"inference binary not found; build with `cmake --build build --config Release`")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?",
                    default="The difference between a CPU and a GPU, in a few sentences, is ")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--bin", type=Path, default=None)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    print(f"[prompt] {args.prompt!r}", file=sys.stderr)
    print(f"[prompt ids] {prompt_ids}", file=sys.stderr)

    binary = find_binary(args.bin)
    cmd = [str(binary),
           "--weights", str(args.weights),
           "--tokens", " ".join(str(t) for t in prompt_ids),
           "--max-tokens", str(args.max_tokens)]
    if args.bench:
        cmd.append("--bench")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)

    out_ids = [int(x) for x in proc.stdout.split() if x.strip()]
    text = tok.decode(out_ids, skip_special_tokens=True)

    print(args.prompt + text)
    print("---", file=sys.stderr)
    sys.stderr.write(proc.stderr)


if __name__ == "__main__":
    main()
