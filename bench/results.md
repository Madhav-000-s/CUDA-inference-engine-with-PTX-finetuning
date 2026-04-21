# Benchmark results

## Setup

- **GPU:** NVIDIA GeForce RTX 5050 Laptop (Blackwell, sm_120, 8 GB)
- **CUDA:** 12.8, driver 580.xx, Windows 11
- **Model:** TinyLlama-1.1B-Chat-v1.0
- **Quantization:** weight-only INT8, symmetric per-output-channel
- **Activations:** FP16, KV cache FP16
- **Decode:** greedy, batch = 1, KV cache enabled
- **Prompt:** `"The difference between a CPU and a GPU, in a few sentences, is "` → 17 input tokens
- **Output length:** 128 tokens

## End-to-end throughput

Mean of 3 runs each, after 2 warmup runs on the HF baseline.

| Run                               | tokens/sec | prefill (ms) | notes                               |
|-----------------------------------|-----------:|-------------:|-------------------------------------|
| HF `transformers` FP16            |      19.14 |         68.1 | cu128 PyTorch nightly, no compile   |
| `tinyllama_cuda` (naive INT8)     |     142.19 |        126.4 | **7.43× vs HF**                     |
| `tinyllama_cuda` (PTX-tuned QKV)  |     140.65 |        117.7 | within noise end-to-end (see below) |

Target was ≥ 2× HF FP16 — cleared by 3.7× on top of that. The hand-tune does
not shift end-to-end tok/s because `qkv_proj` is only ~10–15% of decode time;
attention and the two MLP matmuls dominate. The PTX win is visible at the
kernel level.

## Per-kernel microbench (`bench_qkv.exe`)

Direct timing of `qkv_proj` alone, 50 000 invocations after 100 warmup, QKV
shape = `[2560, 2048]` INT8 weights + `[2048]` FP16 activations (TinyLlama
fused Q+K+V output dim).

| Variant                            | per-call (μs) | weight bw (GB/s) |    speedup |
|------------------------------------|--------------:|-----------------:|-----------:|
| Naive (compiler-generated PTX)     |         25.56 |            204.9 |       1.00× |
| Hand-tuned (inline PTX, f16x2 FMA) |         21.78 |            240.4 |   **1.17×** |

Mean over 5 runs. Naive std: 0.21 μs. Tuned std: 0.36 μs. The per-call gap
(3.78 μs) is ~6× the combined noise, so the speedup is not measurement error.

## PTX hand-tune: `qkv_proj` inner loop

The inner loop accumulates `sum(W_int8[o, :] * x_fp16[:])` for one output row
`o` per block. Each of 256 threads strides 8 columns per iteration. For
`in_dim = 2048` this is exactly one iteration per thread.

### Compiler output (naive) — `build/qkv_proj.ptx`

Per iteration, for each of the 8 columns:

```ptx
ld.global.nc.u8   %rs,  [%rd+offset];    // 1 scalar INT8 load
cvt.s16.s8        %rs,  %rs;
cvt.rn.f32.s16    %f,   %rs;
ld.global.nc.u16  %rs2, [%rd2+offset];   // 1 scalar FP16 load
cvt.f32.f16       %f2,  %rs2;
fma.rn.ftz.f32    %acc, %f2, %f, %acc;   // scalar f32 accumulate
```

Per-iter totals (per thread):

| Instruction class | count |
|-------------------|------:|
| `ld.global.nc.u8`  | 8 |
| `ld.global.nc.u16` | 8 |
| `cvt.s16.s8`       | 8 |
| `cvt.rn.f32.s16`   | 8 |
| `cvt.f32.f16`      | 8 |
| `fma.rn.ftz.f32`   | 8 |

### Hand-tuned — `build/qkv_proj_tuned.ptx`

```ptx
// 8 INT8 weights loaded in one 64-bit transaction
ld.global.nc.v2.u32 {%r_wlo, %r_whi}, [%rd_W];

// 8 FP16 activations loaded in one 128-bit transaction
ld.global.nc.v4.u32 {%r_a0, %r_a1, %r_a2, %r_a3}, [%rd_X];

// Sign-extend each byte to s32 (cvt.s16.s8 is rejected by ptxas 12.8
// on a b32 source, so we widen to s32 then narrow).
cvt.s32.s8   %r_b0, %r_wlo;
shr.u32      %r_t,  %r_wlo, 8;   cvt.s32.s8 %r_b1, %r_t;
shr.u32      %r_t,  %r_wlo, 16;  cvt.s32.s8 %r_b2, %r_t;
shr.u32      %r_t,  %r_wlo, 24;  cvt.s32.s8 %r_b3, %r_t;
// ... same for %r_whi -> b4..b7

// Narrow to f16 and pack pairs into f16x2
cvt.rn.f16.s32  %rs_h0, %r_b0;
cvt.rn.f16.s32  %rs_h1, %r_b1;
mov.b32         %r_w01, {%rs_h0, %rs_h1};   // f16x2 weight pair
// ... same for (h2,h3) -> w_23, (h4,h5) -> w_45, (h6,h7) -> w_67

// Packed FP16x2 FMA into 4 paired accumulators
fma.rn.f16x2  %acc01, %r_w01, %r_a0, %acc01;
fma.rn.f16x2  %acc23, %r_w23, %r_a1, %acc23;
fma.rn.f16x2  %acc45, %r_w45, %r_a2, %acc45;
fma.rn.f16x2  %acc67, %r_w67, %r_a3, %acc67;
```

Per-iter totals (per thread):

| Instruction class     | count | vs naive       |
|-----------------------|------:|----------------|
| `ld.global.nc.v2.u32` |     1 | replaces 8 × u8   |
| `ld.global.nc.v4.u32` |     1 | replaces 8 × u16  |
| `cvt.s32.s8`          |     8 | same count, same cost |
| `cvt.rn.f16.s32`      |     8 | replaces 8 × `cvt.rn.f32.s16` + `cvt.f32.f16` (so 16 → 8 conversions) |
| `mov.b32 {h,h}`       |     4 | new (register pack) |
| `fma.rn.f16x2`        |     4 | replaces 8 × `fma.rn.ftz.f32` |

Net: 16 scalar global loads → 2 wide loads, 16 conversions → 8, 8 scalar
FMAs → 4 packed FMAs. That's the shape the 17% kernel speedup falls out of.

The f16x2 accumulator is safe here: four packed FMAs per thread each add a
value bounded by `|s8| · |fp16 activation| ≈ 127 · O(1)`, so partial sums
stay inside FP16 range. A final f32 promotion happens once before the warp
reduction.

### Register / smem usage (`ptxas -v`)

| Variant | registers | smem (bytes) | barriers |
|---------|----------:|-------------:|---------:|
| Naive   |        39 |           32 |        1 |
| Tuned   |        40 |           32 |        1 |

No spills in either. +1 register cost for the packed accumulator set.

## Why the end-to-end number didn't move

`qkv_proj` runs once per decoded token per layer (22 layers × 1 call ≈ 22
invocations/token at ~25 μs each → 0.55 ms/token). But `attention`,
`matmul_int8` for `o_proj`, and the two MLP projections together dominate the
remaining ~6.5 ms/token. Shaving 4 μs off each `qkv_proj` call nets ~88 μs
per decoded token, which is the same order of magnitude as run-to-run
variance (~10 ms across a 128-token decode).

To turn the hand-tune into a visible end-to-end win, the same inline-PTX
pattern would need to be applied to `o_proj` and the two MLP matmuls (they
share the same INT8 × FP16 inner loop). That's the obvious next step; it was
out of scope for this milestone, which was specifically "one hand-tuned PTX
hot path with before/after evidence."

## Note on `ncu`

Nsight Compute 2025.1.0 (shipped with CUDA 12.8) does **not** recognize
GB207 (RTX 5050 Laptop) — `ncu --list-chips` enumerates only
`gb202, gb203, gb205`, and profiling the binary returns
`LibraryNotLoaded. Check that a compatible driver library is loaded`.
Nsight Compute 2025.3+ is required for the laptop Blackwell chips. The
per-kernel CUDA-event timing above substitutes for the SM/DRAM throughput
panels that `ncu --set full` would normally report.
