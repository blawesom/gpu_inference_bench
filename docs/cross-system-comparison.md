# Cross-System Performance Comparison

gpu_inference_bench — 3 systems, identical workload (random 512-in/256-out tokens, 50 prompts, seed 42, temperature 0, C = 1/4/8/16), vLLM v0.28.0, 3-model matrix.

## Systems compared

| Field | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| GPU | 0x7551 (31.9 GB) | Intel(R) Arc(TM) Pro B70 Graphics (31.9 GB) | NVIDIA L40 (45.0 GB) |
| Stack | rocm 7.2.53211, aiter unknown | n/a | cuda 13.0 |
| Driver | 7.2.4 | n/a | 610.57.04 |
| vLLM | 0.28.0+rocm723 | 0.28.0+xpu | 0.28.0 |
| Image | vllm/vllm-openai-rocm:v0.28.0 | vllm/vllm-openai-xpu:v0.28.0 | vllm/vllm-openai:v0.28.0 |
| OS / CPU | CachyOS / AMD Ryzen 7 9800X3D 8-Core Processor | CachyOS / AMD Ryzen 7 9800X3D 8-Core Processor | Rocky Linux 10.2 (Red Quartz) / Intel Xeon Processor (SapphireRapids) |
| Run dir (re-run) | `results/20260913-141852_0x7551` | `results/20260909-210037_intel-r-arc-tm-pro-b70-graphics` | `results/20260909-142627_nvidia-l40` |
| Run dir (orig.) | `results/20260903-222650_0x7551` | `results/20260904-212058_intel-r-arc-tm-pro-b70-graphics` | `results/20260905-134350_nvidia-l40` |

## Executive summary

- **NVIDIA L40 leads output throughput on all three models** at C=16 baseline (peak 706 tok/s); across the 3-model matrix Intel Arc Pro B70 averages 71% and AMD 0x7551 47% of the leader's throughput.
- **NVIDIA L40 is also the most power-efficient**: 1.71 mean output tok/s per watt @ C=16 vs 1.45 (Intel Arc Pro B70) and 0.81 (AMD 0x7551) — 2.1× the slowest. All three systems use measured power in the aligned window.
- **NVIDIA L40 has the lowest decode latency**: mean TPOT p50 23 ms vs 60 ms for AMD 0x7551; tails (mean p99/p50) are tightest on AMD 0x7551 (1.4×) and loosest on Intel Arc Pro B70 (1.6×).

## Performance

### Peak output throughput @ C=16, baseline (tok/s; % of row best)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 277.8 (68%) | 345.8 (85%) | 406.2 (100%) |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 93.1 (33%) | 183.0 (66%) | 278.7 (100%) |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 273.2 (39%) | 431.7 (61%) | 706.1 (100%) |

### Batch scaling, C=1 → C=16 (baseline throughput ratio)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 14.7× | 10.1× | 9.3× |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 3.3× | 6.8× | 7.3× |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 16.4× | 9.3× | 6.1× |

### Takeaways

- The cross-system gap is **narrowest on M1** (dense ~9B, BF16: last place still at 68% of best) and **widest on M3** (dense 27B, AWQ-4bit: 33%).
- **AMD 0x7551 barely scales on M3**: 3.3× from C=1 to C=16 (28 → 93 tok/s) while TPOT p50 climbs 34 → 144 ms — batch decode degrades under load (KV/scheduling pressure on the tight 32 GB fit and/or a ROCm batching inefficiency for this checkpoint).
- **AMD 0x7551 M4 single-stream anomaly**: 16.7 tok/s at C=1 vs 147.6 at C=4 (9× step) — single-stream MoE decode is inefficient on this stack; the 16.4× C=1→C=16 'scaling' partly reflects this, not superlinear batching.
- **long-context (32k max-model-len) is a no-op for M1** throughput (only model with that cell): Intel Arc Pro B70 -0.0%, NVIDIA L40 -0.4% vs baseline @ C=16 — expected, since the workload still sends 512-token prompts and only the supported context window grows (AMD's M1 long-context run failed at engine startup; see caveats).

## Latency (ms)

Averages over all baseline cells (3 models × C=1/4/8/16). **p50** = typical request, **p99** = worst 1% of requests. TTFT = time to first token (prefill + queueing); TPOT = per-token decode latency; ITL = inter-token gap (streaming tail risk).

| Metric | AMD 0x7551 p50 | AMD 0x7551 p99 | Intel Arc Pro B70 p50 | Intel Arc Pro B70 p99 | NVIDIA L40 p50 | NVIDIA L40 p99 |
|---|---|---|---|---|---|---|
| TTFT | 917 | 1438 | 805 | 1243 | 684 | 980 |
| TPOT | 60 | 64 | 33 | 36 | 23 | 25 |
| ITL | 59 | 93 | 32 | 67 | 22 | 45 |

### Takeaways

- **TTFT**: single-stream prefill is fast everywhere (C=1 p50 up to 478 ms); under full load (C=16) NVIDIA L40 queues fastest (1255 ms mean p50 across models) vs 1673 ms for AMD 0x7551 — the heavy M3 dense-27B prefill dominates (2.5–3.6 s p50 per request at C=16).
- **TPOT**: NVIDIA L40 decodes fastest at every concurrency level (23 ms mean p50 vs 60 ms for AMD 0x7551). M1 @ C=16 spans 1.6× across systems (29 → 47 ms).
- **ITL tails**: the worst single cell is AMD 0x7551 on cyankiwi/Qwen3.8-27B-AWQ-INT4 @ C=16 (491 ms p99) — streaming stalls of ~0.5 s. In absolute terms NVIDIA L40 still keeps the worst 1% of inter-token gaps lowest (45 ms vs 93 ms for AMD 0x7551); relative tail width (mean p99/p50) is tightest on AMD 0x7551 (1.4×).

## Power efficiency (output tok/s per watt)

C=16, baseline. AMD 0x7551: aligned to the measured window, measured · Intel Arc Pro B70: aligned to the measured window, measured · NVIDIA L40: aligned to the measured window, measured. Energy is GPU-only (vendor power sensor), not system energy; `energy_j_per_ktok` is per level.

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 1.19 (233 W) | 1.53 (226 W) | 1.43 (284 W) |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 0.31 (299 W) | 0.82 (222 W) | 0.94 (298 W) |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 0.93 (294 W) | 1.98 (218 W) | 2.77 (255 W) |
| **Overall (mean)** | **0.81** | **1.45** | **1.71** |

### Takeaways

- **NVIDIA L40 is 2.1× AMD 0x7551 overall.** The gap is largest on M4 (MoE 35B): 2.77 vs 0.93 tok/s/W — strong MoE execution at a moderate 255 W — and smallest on M1: 1.43 vs 1.19.
- Worst efficiency cell: AMD 0x7551 on cyankiwi/Qwen3.8-27B-AWQ-INT4 (0.31 tok/s/W at 299 W) — dense 27B decode is power-hungry on this stack.

## Protocol change: original (Sept 3–5) → re-run (corrected)

The original runs used the legacy protocol (prefix caching **ON**, single measured pass, power over the full bench-client window, no Intel power telemetry). The re-run applies the 2026-09-08 fixes: prefix caching **OFF** (verified per cell), per-level warmup until no JIT, **3-pass median**, power/energy **aligned to the measured window**, and **measured Intel power** (xpu-smi 2.1.0). Two fixes oppose each other on throughput — prefix-caching off *removes replay cache hits* (deflates) while warmup-until-no-JIT *removes in-bench Triton JIT stalls* (inflates) — and they roughly cancel, except where the JIT stall was the dominant legacy defect.

### C=16 baseline output throughput, original → re-run (tok/s, Δ%)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 174 → 278 (+60%) | 345 → 346 (+0%) | 407 → 406 (-0%) |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 88 → 93 (+5%) | 182 → 183 (+0%) | 279 → 279 (-0%) |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 272 → 273 (+0%) | 425 → 432 (+2%) | 706 → 706 (-0%) |
### Mean power efficiency (tok/s/W @ C=16, rankable models), original → re-run

| Card | original | re-run | Δ | power (orig → re-run) |
|---|---|---|---|---|
| AMD 0x7551 | 0.72 | 0.81 | +12% | measured → measured |
| Intel Arc Pro B70 | 1.38 | 1.45 | +5% | assumed 230 W → measured |
| NVIDIA L40 | 2.14 | 1.71 | -20% | measured → measured |

### Share of leader @ C=16 (rankable models), original → re-run

| Model | AMD 0x7551 old → new | Intel Arc Pro B70 old → new | NVIDIA L40 old → new |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 43% → 68% | 85% → 85% | 100% → 100% |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 32% → 33% | 65% → 66% | 100% → 100% |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 39% → 39% | 60% → 61% | 100% → 100% |

### Ranking impact

- **Throughput order** unchanged: NVIDIA L40 > Intel Arc Pro B70 > AMD 0x7551 → NVIDIA L40 > Intel Arc Pro B70 > AMD 0x7551.
- **Efficiency order** unchanged: NVIDIA L40 > Intel Arc Pro B70 > AMD 0x7551 → NVIDIA L40 > Intel Arc Pro B70 > AMD 0x7551.
- **NVIDIA L40’s efficiency lead over AMD 0x7551** shrank from 3.0× (original) to **2.1×** (re-run): the legacy power window under-stated steady-state draw, so the original figures were optimistic.
- The largest single-cell throughput swing is **AMD 0x7551 M1** (+60% at C=16 baseline) — the rest of the matrix moves <5%, so the gap compression is driven by that one JIT-deflated cell.
- **Intel Arc Pro B70** is now on **measured** power (xpu-smi 2.1.0) instead of an assumed 230 W TDP floor — its efficiency is real draw, so the old “conservative floor” caveat no longer applies.

## Caveats

- **Protocol (P0/P1/T1) now enforced**: prefix caching OFF (verified per cell), per-level warmup until no JIT, 3-pass median, and power/energy aligned to the measured bench window (GPU-only, vendor power sensor). The Sept 3–5 legacy runs (prefix caching ON, single pass, unaligned power window, no Intel power telemetry) are **superseded** — see the "Protocol change" section for the effect on their numbers.
- **AMD 0x7551 re-run**: **failed cells**: long-context (pass1-failed:engine-startup); **single-pass (not a 3-pass median)**: Qwen3.8-27B-AWQ-INT4.
- **AITER (T1) M1 on AMD**: the aiter-attn cell failed at engine startup (`pass1-failed:engine-startup`) — no AITER throughput was captured this run, so the ROCm Triton-fallback baseline remains the AMD attention reference. AITER is AMD-only; the cell auto-skips on NVIDIA/Intel.
- **VRAM differs**: NVIDIA L40 has 45 GB vs 32 GB on AMD/Intel. All three models fit at this workload (~12.3 k KV tokens at C=16); the extra headroom only matters for long-context cells.
- **Host CPUs differ**: AMD/Intel runs on an AMD Ryzen 7 9800X3D (consumer), NVIDIA on an Intel Xeon (Sapphire Rapids). Negligible for GPU-bound decode; noted for completeness.