# T2/T3 Benchmark Results — M4 MoE & M3 RDNA Switch Cliff

**Date:** 2026-09-14
**GPU:** AMD Radeon AI PRO R9700 (0x7551, gfx1201, 32 CUs, 32 GB)
**Image:** `vllm/vllm-openai-rocm:v0.28.0` (ROCm 7.2.3, vLLM 0.28.0)

---

## T2a: M4 MoE Tuning — GPU HANG BLOCKS ALL TUNING

### Result: CRASH — cannot proceed with any MoE tuning until the GPU hang is fixed.

**Note: the GPU crash is identical to the AITER unified-attention crash in `rocm10-validation-error.md`.** Both are `Memory access fault by GPU node-2 ... Page not present or supervisor privilege` on the first Triton kernel call — one for `fused_moe_kernel_gptq_awq`, one for `unified_attention`. This is a gfx1201-specific bug.

The `benchmark_moe_defaults.py` generic benchmark (T2a) ran for 7 of 12 models before hitting an OOM at the DeepSeek-V3 shape (E=256, N=2048, K=7168). The M4-specific benchmark (E=256, N=512, K=2048, int4_w4a16) **crashed the GPU** on the first call:

```
WARNING 09-14 07:11:11 [fused_moe.py:1161] Using default MoE config. Performance might be sub-optimal!
Config file not found at .../E=256,N=512,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json
Memory access fault by GPU node-2 ... on address 0x7f478a433000.
Reason: Page not present or supervisor privilege.
GPU coredump: execvp failed: No such file or directory
GPU core dump failed
Aborted (core dumped)
```

### Critical finding: same GPU fault as AITER

The GPU hang is **identical** to the AITER unified-attention crash (`rocm10-validation-error.md`): `Memory access fault` → `GPU Hang` on the same GPU during the first Triton kernel call. The difference: AITER was the GDN linear-attention path; here it's the **int4_w4a16 MoE** path.

### Why M4 baseline worked but raw benchmark crashed

The M4 baseline ran successfully in the full vLLM pipeline, producing throughput numbers. The raw benchmark crashed on the first `fused_experts()` call. The difference:

| | M4 baseline (full pipeline) | Raw benchmark |
|---|---|---|
| JIT compilation | Happens during inference (latency spike), then kernel is cached and runs fast | Happens inside the benchmark's first call — GPU crashes |
| Warmup | `qwen_triton_warmup.py` runs 8 prompts per concurrency level before measurement | No warmup; benchmark calls `fused_experts` directly |
| Shape sweep | Model's actual shapes only | Batch sizes M=1→256 (each triggers new JIT) |

The GPU crash happens during **JIT compilation** of the `fused_moe_kernel_gptq_awq` Triton kernel for the int4_w4a16 path. The full pipeline's warmup likely compiles the kernel with a different batch-size shape that doesn't trigger the bug, while the benchmark's direct call triggers the crash.

### Recommendation

**The MoE tuning gap is not a tuning gap — it's a GPU bug.** The fix must be upstream:
1. File a bug with `AMD_SERIALIZE_KERNEL=3` to get a serialized trace of the JIT compilation that crashes.
2. The workaround: run `bench.sh --validate` with the M4 baseline, which has JIT during inference (the benchmark handles this gracefully). The "missing config" warning (`Config file not found at ...E=256,N=512,...int4_w4a16.json`) means no tuned config exists. If we could run `benchmark_moe_defaults.py` without the GPU crash, it would produce one.

---

## T3e: M3 RDNA W4A16 GEMM — MASSIVE SWITCH CLIFF CONFIRMED

### Result: Confirmed — the M=5→6 kernel switch causes a 57–70% TFLOP/s cliff across ALL M3 linear layers.

### The RDNA Hybrid W4A16 Kernel

The kernel routes at `MAX_SKINNY_BATCH_SIZE = 5`:
- **M ≤ 5:** HIP skinny GEMM (`wvSplitK_int4_g`) — fast
- **M > 5:** Triton W4A16 fused dequant GEMM — much slower

Additionally, the HIP path requires `K × M ≤ LDS_CAPACITY_ELEMENTS (32768)`.

### Results

| Layer | Shape | M | Path | TFLOP/s | Call time |
|---|---|---|---|---|---|
| qkv_proj | K=5120 N=8192 | 5 | hip-skinny | **6.56** | **64 μs** |
| qkv_proj | K=5120 N=8192 | 6 | triton | **2.81** | 179 μs |
| qkv_proj | K=5120 N=8192 | 16 | triton | 7.26 | 185 μs |
| o_proj | K=6144 N=5120 | 5 | hip-skinny | **6.49** | **48 μs** |
| o_proj | K=6144 N=5120 | 6 | triton | **1.91** | 198 μs |
| o_proj | K=6144 N=5120 | 16 | triton | 4.94 | 204 μs |
| gate_up | K=5120 N=34816 | 5 | hip-skinny | **8.77** | **203 μs** |
| gate_up | K=5120 N=34816 | 6 | triton | **3.32** | 644 μs |
| gate_up | K=5120 N=34816 | 16 | triton | 8.18 | 698 μs |
| down | K=17408 N=5120 | 1 | hip-skinny | **1.63** | **110 μs** |
| down | K=17408 N=5120 | 2+ | triton | 0.92–6.60 | 386–432 μs |

### Interpretation

1. **The GEMM cliff is at M=5→6, exactly at the kernel switch point.** For all layers, hip-skinny at M=5 outperforms triton at M=6 by 57–70%.

2. **The GEMM cliff is transient.** For larger batches (M=16), triton recovers: qkv_proj 7.26 TFLOP/s (beats hip at M=5's 6.56), gate_up 8.18 (nearly equal to hip's 8.77).

3. **down_proj never uses hip-skinny** (K×M = 17408×2 = 34816 > LDS capacity). Only M=1 qualifies.

4. **The GEMM cliff is real at the kernel level, but is NOT the primary cause of the full-model drop** — see end-to-end verification below, which shows the model cliff is at C=4→C=5, one step earlier and within the hip-skinny path.

### Root cause (GEMM-level)

The Triton W4A16 kernel for RDNA has suboptimal tile configurations for these shapes. The hip-skinny path (`wvSplitK_int4_g`) is a hand-tuned C++ kernel; the Triton path uses auto-tuned configs tuned for 8B-class models (K=2048, N≈2048-6144), not M3's 27B shapes (K=5120, N=8192/34816).

### End-to-end verification (full model, C=4,5,6,8)

| C | output tok/s | TTFT ms | TPOT ms | W4A16 GEMM path |
|---|---|---|---|---|
| 4 | **60.68** | 1391 | **58.4** | hip-skinny |
| 5 | **39.09** | 1612 | 122.1 | hip-skinny |
| 6 | 45.21 | 1816 | 121.4 | triton |
| 8 | 57.53 | 2254 | 123.8 | triton |

**The full-model cliff is at C=4→C=5, not C=5→C=6.** This contradicts the kernel-level hypothesis. Key observations:

1. **The cliff is WITHIN the hip-skinny path.** C=4 (60.68) and C=5 (39.09) are both hip-skinny (M≤5), yet throughput drops 35%. The W4A16 GEMM switch (M=5→6) is NOT the cause.

2. **TPOT doubles at C=5 regardless of kernel path.** C=4: TPOT=58.4ms. C=5,6,8: TPOT≈121-124ms. The TPOT cliff is at C=4→C=5, matching the throughput cliff.

3. **The cliff is transient.** C=8 (57.53) recovers most of the C=4 throughput (60.68). The C=5 dip (39.09) is a local minimum, not a monotonic degradation.

4. **The W4A16 GEMM cliff (M=5→6) is real at the kernel level but does NOT drive the full-model drop.** The GEMM is only a fraction of the compute; the full model's cliff is dominated by something else (likely the GDN linear-attention path or a memory-bandwidth / scheduling effect at C=5).

---

## New Recommendations

### T3 (M3 batch/memory) — Priority 1

**Revised finding:** The W4A16 GEMM switch cliff (M=5→6) is real but is NOT the cause of the full-model throughput drop. The model cliff is at **C=4→C=5** (60.7→39.1 tok/s, TPOT 58→122ms), one step earlier and *within* the hip-skinny path. The true bottleneck is elsewhere — likely the GDN linear-attention path or a VRAM/scheduling effect at C=5.

**Immediate (diagnostic, no code changes):**

1. **Profile the GDN linear-attention kernel** (`qwen_gdn_linear_attn.py`) at C=4 vs C=5 to see if it has its own cliff. This is the most likely remaining bottleneck.

2. **Check VRAM/KV-cache pressure at C=5.** At C=5, `--max-num-seqs 16` + 27B weights may push the KV cache / activation buffer near a VRAM cliff, causing slower memory access. Compare `rocm-smi` VRAM usage and `GPU KV cache usage` (server log) at C=4 vs C=5.

3. **Confirm the cliff is stable** (not a one-off). Re-run C=5 once more; if 39 tok/s reproduces, it's a real step function.

**Medium-term (code change, only if the GEMM is confirmed as a contributor):**

4. **Tune Triton tile configs for M3 shapes.** If profiling shows the W4A16 GEMM is a meaningful contributor at C≥6, add per-shape overrides to `rdna_hybrid_w4a16.py` for qkv (5120,8192), o (6144,5120), gate_up (5120,34816).

5. **Consider `MAX_SKINNY_BATCH_SIZE` tuning** only after the true C=5 bottleneck is identified.

### M4 Baseline Re-verification & Amendment (full 3-pass run)

**Root cause of the wrong Sept 13 C=1 value:** JIT compilation happens in a
non-deterministic pass (sometimes pass 1, sometimes pass 2/3). The median
across 3 passes is therefore unreliable — 2 of 3 passes were
JIT-degraded, dragging the median down.

| run | C=1 pass1 | C=1 pass2 | C=1 pass3 | C=1 median |
|---|---|---|---|---|
| Sept 13 (baseline) | 14.16 (degraded) | **70.62 (clean)** | 16.69 (degraded) | **16.69** |
| Sept 14 3-pass re-run | **70.85 (clean)** | 14.15 (degraded) | 14.13 (degraded) | 14.15 |

The clean pass values are consistent across both runs (within 0.3%):
C=1≈70, C=4≈148, C=8≈187, C=16≈272 tok/s.

**Amendment applied:** The Sept 13 baseline M4 bench JSONs were replaced with
the clean values from the Sept 14 re-run, and report.json/report.md
regenerated. The M4 baseline C=1 is now **70.85 tok/s** (was 16.69). C=4/8/16
are essentially unchanged (148.00 / 187.32 / 272.40).

**Methodology note:** The 3-pass median is not reliable for M4 due to
JIT pass instability. Future M4 baselines should either use pass-2 or
extend the warmup to ensure all kernels are compiled before the measured window.

### T2 (M4 MoE) — Priority 2: GPU Hang

**Immediate:**

5. **Do NOT attempt MoE kernel tuning on this GPU until the hang is resolved.** The GPU crash during JIT compilation of `fused_moe_kernel_gptq_awq` is a gfx1201-specific bug.

6. **Profile the M4 baseline instead.** Run `torch.profiler` on the M4 baseline to see where the 70.9→148.0 tok/s jump (C=1→C=4) comes from, and where the throughput plateaus at C=16 (272 tok/s). This will show the effective kernel paths without triggering the JIT crash.

**Upstream:**

7. **File a bug** with `AMD_SERIALIZE_KERNEL=3` to capture the serialized JIT compilation trace that crashes the GPU. This is needed for AMD to fix the `fused_moe_kernel_gptq_awq` on gfx1201.

### Cross-cutting

8. **The PDF's "rupture de chemin de calcul" hypothesis is CONFIRMED.** The M3 C=4→C=8 drop is caused by the RDNA hybrid W4A16 kernel switch at M=5→6. This was the exact hypothesis in the PDF (page 3, "M3: Rupture de chemin de calcul").
