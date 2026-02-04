# Benchmark results

Populated after the engine is built and run. Sections are placeholders until
then.

## Setup

- GPU: NVIDIA GeForce RTX 5050 Laptop (Blackwell, sm_120, 8 GB)
- Model: TinyLlama-1.1B-Chat-v1.0
- Prompt: see `scripts/baseline_benchmark.py` (32-token fixed prompt)
- Output length: 128 tokens, greedy

## Throughput

| Run                          | tokens/sec | first-token ms | peak VRAM |
|------------------------------|-----------:|---------------:|----------:|
| HF `transformers` FP16       |        TBD |            TBD |       TBD |
| tinyllama_cuda (naive INT8)  |        TBD |            TBD |       TBD |
| tinyllama_cuda (PTX tuned)   |        TBD |            TBD |       TBD |

Target: naive INT8 ≥ 2× HF FP16. PTX tuned uplifts from there.

## PTX hand-tune: `qkv_proj` inner loop

### Compiler output (naive)

```
(populate with the inner-loop snippet from build/qkv_proj.ptx after
 `cmake --build build --target qkv_ptx`)
```

### Hand-tuned

```
(populate with the inline-asm PTX diff)
```

### ncu metrics

| Metric                                                | naive | tuned |
|-------------------------------------------------------|------:|------:|
| sm__throughput.avg.pct_of_peak_sustained_elapsed      |   TBD |   TBD |
| dram__throughput.avg.pct_of_peak_sustained_elapsed    |   TBD |   TBD |
| smsp__inst_executed.sum                               |   TBD |   TBD |
| achieved_occupancy                                    |   TBD |   TBD |

Command:
```
ncu --set full --kernel-name regex:qkv_proj_kernel --launch-count 1 \
    build/Release/inference.exe --weights weights/tinyllama-int8.bin \
    --tokens "1 2 3" --max-tokens 4
```
