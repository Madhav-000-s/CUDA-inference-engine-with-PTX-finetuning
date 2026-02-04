# TinyLlama Quantized Inference Engine (CUDA + PTX)

From-scratch INT8 inference engine for **TinyLlama-1.1B-Chat-v1.0**, written in C++/CUDA and targeting an **RTX 5050 Laptop** (Blackwell, `sm_120`).

Portfolio goals:
- Custom CUDA kernels for the full Llama forward pass (RMSNorm, fused QKV, tiled SDPA with GQA, SwiGLU, LM head).
- **Weight-only INT8** (symmetric, per-output-channel) with FP16 activations.
- **≥ 2× throughput** vs HuggingFace `transformers` FP16 baseline on the same GPU.
- One hot path **hand-tuned in inline PTX** with `nsys`/`ncu` before-after evidence.

## Prerequisites (Windows)

1. **Visual Studio 2022 Build Tools** with "Desktop development with C++" workload.
2. **CUDA Toolkit 12.8 or newer** (first release supporting `sm_120`).
3. **CMake 3.24+** (bundled with VS installer, or stand-alone).
4. **Python 3.10+** with:
   ```
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
   pip install transformers sentencepiece numpy
   ```

Verify:
```
nvcc --version          # → 12.8.x
cmake --version         # → 3.24+
python -c "import torch; print(torch.cuda.get_device_capability())"   # → (12, 0)
```

## Build

```
cd "CUDA inference engine with PTX fine-tuning"
python scripts/export_weights.py                 # writes weights/tinyllama-int8.bin
python scripts/baseline_benchmark.py             # records HF FP16 tokens/sec

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
./build/Release/inference.exe --prompt "Hello, my name is" --max-tokens 128 --bench
```

## Repository layout

```
scripts/        Python: weight export + HF baseline benchmark
include/        C++ headers (config, tensor, weight loader)
src/            C++ host code
src/kernels/    CUDA kernels  (the PTX target is src/kernels/qkv_proj.cu)
bench/          nsys/ncu results, PTX before/after snippets
```

See `bench/results.md` for throughput numbers and PTX hand-tune evidence.
