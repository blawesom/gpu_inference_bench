# Cross-System Performance Comparison

gpu_inference_bench — 3 systems, identical workload (random 512-in/256-out tokens, 50 prompts, seed 42, temperature 0, C = 1/4/8/16), vLLM v0.28.0, 4-model matrix.

## Systems compared

| Field | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| GPU | 0x7551 (31.9 GB) | Intel(R) Arc(TM) Pro B70 Graphics (31.9 GB) | NVIDIA L40 (45.0 GB) |
| Stack | rocm 7.2.53211 | n/a | cuda 13.0 |
| Driver | 7.2.2 | n/a | 610.57.04 |
| vLLM | 0.28.0+rocm723 | 0.28.0+xpu | 0.28.0 |
| Image | vllm/vllm-openai-rocm:v0.28.0 | vllm/vllm-openai-xpu:v0.28.0 | vllm/vllm-openai:v0.28.0 |
| OS / CPU | CachyOS / AMD Ryzen 7 9800X3D 8-Core Processor | CachyOS / AMD Ryzen 7 9800X3D 8-Core Processor | Rocky Linux 10.2 (Red Quartz) / Intel Xeon Processor (SapphireRapids) |
| Run dir | `results/20260903-222650_0x7551` | `results/20260904-212058_intel-r-arc-tm-pro-b70-graphics` | `results/20260905-134350_nvidia-l40` |

## Executive summary

- **NVIDIA L40 leads output throughput on all four models** at C=16 baseline (peak 706 tok/s); across the 4-model matrix Intel Arc Pro B70 averages 74% and AMD 0x7551 44% of the leader's throughput.
- **NVIDIA L40 is also the most power-efficient**: 2.00 mean output tok/s per watt @ C=16 vs 1.28 (Intel Arc Pro B70) and 0.75 (AMD 0x7551) — 2.7× the slowest. Intel's figure is a conservative floor (230 W assumed, no GPU telemetry captured).
- **NVIDIA L40 has the lowest decode latency**: mean TPOT p50 27 ms vs 72 ms for AMD 0x7551; tails (mean p99/p50) are tightest on Intel Arc Pro B70 (1.8×) and loosest on NVIDIA L40 (1.9×).
- **kv-fp8 KV cache is neutral-to-negative on every system** (mean Δ vs baseline @ C=16: -4.9% AMD 0x7551, -2.2% Intel Arc Pro B70, -0.5% NVIDIA L40) — the KV cache is not the bottleneck at this workload size (~12.3 k KV tokens at C=16).

## Performance

### Peak output throughput @ C=16, baseline (tok/s; % of row best)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 174.1 (43%) | 344.5 (85%) | 406.6 (100%) |
| M2 · openai/gpt-oss-20b | 171.6 (63%) | 228.0 (84%) | 270.8 (100%) |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 88.5 (32%) | 182.4 (65%) | 278.9 (100%) |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 272.2 (39%) | 424.8 (60%) | 706.2 (100%) |

### Batch scaling, C=1 → C=16 (baseline throughput ratio)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 9.4× | 10.1× | 9.3× |
| M2 · openai/gpt-oss-20b | 10.6× | 7.7× | 6.3× |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 3.1× | 6.8× | 7.3× |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 18.3× | 9.8× | 6.1× |

### kv-fp8 vs baseline @ C=16 (output throughput %)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | -3.0% | -0.6% | +1.3% |
| M2 · openai/gpt-oss-20b | -12.9% | -6.8% | -4.1% |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | +3.5% | -0.9% | +0.8% |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | -7.2% | -0.4% | +0.1% |

### Takeaways

- The cross-system gap is **narrowest on M2** (MoE 21B/3.6B active, MXFP4: last place still at 63% of best) and **widest on M3** (dense 27B, AWQ-4bit: 32%).
- **AMD 0x7551 barely scales on M3**: 3.1× from C=1 to C=16 (28 → 88 tok/s) while TPOT p50 climbs 34 → 157 ms — batch decode degrades under load (KV/scheduling pressure on the tight 32 GB fit and/or a ROCm batching inefficiency for this checkpoint).
- **AMD 0x7551 M4 single-stream anomaly**: 14.8 tok/s at C=1 vs 146.8 at C=4 (10× step) — single-stream MoE decode is inefficient on this stack; the 18.3× C=1→C=16 'scaling' partly reflects this, not superlinear batching.
- kv-fp8 hurts **AMD 0x7551** most (mean -4.9%), worst cell M2 -12.9% @ C=16. Single-stream is hit even harder: M3 on AMD 0x7551 -49% @ C=1. No system benefits at this workload size.
- **long-context (32k max-model-len) is a no-op for M1** throughput (only model with that cell): Intel Arc Pro B70 +0.1%, NVIDIA L40 -0.5% vs baseline @ C=16 — expected, since the workload still sends 512-token prompts and only the supported context window grows (AMD's M1 long-context run failed at engine startup; see caveats).

## Latency (ms)

Averages over all baseline cells (4 models × C=1/4/8/16). **p50** = typical request, **p99** = worst 1% of requests. TTFT = time to first token (prefill + queueing); TPOT = per-token decode latency; ITL = inter-token gap (streaming tail risk).

| Metric | AMD 0x7551 p50 | AMD 0x7551 p99 | Intel Arc Pro B70 p50 | Intel Arc Pro B70 p99 | NVIDIA L40 p50 | NVIDIA L40 p99 |
|---|---|---|---|---|---|---|
| TTFT | 665 | 1142 | 608 | 955 | 497 | 799 |
| TPOT | 72 | 96 | 38 | 54 | 27 | 37 |
| ITL | 55 | 144 | 28 | 67 | 19 | 55 |

### Takeaways

- **TTFT**: single-stream prefill is fast everywhere (C=1 p50 up to 465 ms); under full load (C=16) NVIDIA L40 queues fastest (862 ms mean p50 across models) vs 1180 ms for Intel Arc Pro B70 — the heavy M3 dense-27B prefill dominates (2.3–3.4 s p50 per request at C=16).
- **TPOT**: NVIDIA L40 decodes fastest at every concurrency level (27 ms mean p50 vs 72 ms for AMD 0x7551). M1 @ C=16 spans 2.5× across systems (29 → 74 ms).
- **ITL tails**: the worst single cell is AMD 0x7551 on cyankiwi/Qwen3.8-27B-AWQ-INT4 @ C=16 (1172 ms p99) — streaming stalls of ~1.2 s. In absolute terms NVIDIA L40 still keeps the worst 1% of inter-token gaps lowest (55 ms vs 144 ms for AMD 0x7551); relative tail width (mean p99/p50) is tightest on Intel Arc Pro B70 (1.8×).

## Power efficiency (output tok/s per watt)

C=16, baseline. AMD/NVIDIA: measured `power_avg_w` over the bench run. Intel: no GPU telemetry captured → evaluated at the documented **230 W max TDP**, so its values are conservative floors.

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 0.83 (210 W) | 1.50 (230 W*) | 1.72 (237 W) |
| M2 · openai/gpt-oss-20b | 0.82 (209 W) | 0.99 (230 W*) | 1.56 (174 W) |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 0.31 (289 W) | 0.79 (230 W*) | 1.08 (257 W) |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 1.04 (263 W) | 1.85 (230 W*) | 3.63 (194 W) |
| **Overall (mean)** | **0.75** | **1.28** | **2.00** |

Note: Intel power assumed (max TDP), not measured.

### Takeaways

- **NVIDIA L40 is 2.7× AMD 0x7551 overall.** The gap is largest on M4 (MoE 35B): 3.63 vs 1.04 tok/s/W — strong MoE execution at a moderate 194 W — and smallest on M1: 1.72 vs 0.83.
- Worst efficiency cell: AMD 0x7551 on cyankiwi/Qwen3.8-27B-AWQ-INT4 (0.31 tok/s/W at 289 W) — dense 27B decode is power-hungry on this stack.
- Intel Arc Pro B70's true efficiency is at or above the values shown (assumed full TDP; actual draw was likely lower).

## Caveats

- **VRAM differs**: NVIDIA L40 has 45 GB vs 32 GB on AMD/Intel. All four models fit comfortably on 32 GB at this workload (~12.3 k KV tokens at C=16), so the extra headroom does not change scheduling; it only matters for long-context cells.
- **AMD**: M1 `long-context` cell failed at engine startup (`failed: engine-startup`); shown as n/a. All other cells ran.
- **Intel**: the XPU run captured **no GPU telemetry** (mem/util/power). Power-efficiency figures for Intel assume the documented **max TDP of 230 W** for the Arc Pro B70 (Intel Arc Pro B-Series spec sheet) — an upper bound on actual draw, so Intel's tok/s-per-W values are conservative floors (true efficiency is at or better than shown).
- **Host CPUs differ**: AMD/Intel runs on an AMD Ryzen 7 9800X3D (consumer), NVIDIA on an Intel Xeon (Sapphire Rapids). Negligible for GPU-bound decode; noted for completeness. The AMD run used GPU index 1 on a dual-GPU host.