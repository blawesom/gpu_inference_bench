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

- **NVIDIA L40 leads output throughput on most models** at C=16 baseline (peak 706 tok/s); across the 3-model matrix (M2 excluded: output-token shortfall, see † below) Intel Arc Pro B70 averages 70% and AMD 0x7551 38% of the leader's throughput.
- **NVIDIA L40 is also the most power-efficient**: 2.14 mean output tok/s per watt @ C=16 vs 1.38 (Intel Arc Pro B70) and 0.72 (AMD 0x7551) — 3.0× the slowest. Intel's figure is a conservative floor (230 W assumed, no GPU telemetry captured).
- **NVIDIA L40 has the lowest decode latency**: mean TPOT p50 24 ms vs 68 ms for AMD 0x7551; tails (mean p99/p50) are tightest on Intel Arc Pro B70 (1.6×) and loosest on NVIDIA L40 (1.8×).
- **kv-fp8 KV cache is neutral-to-negative on every system** (mean Δ vs baseline @ C=16: -2.2% AMD 0x7551, -0.7% Intel Arc Pro B70, +0.7% NVIDIA L40) — the KV cache is not the bottleneck at this workload size (~12.3 k KV tokens at C=16).

## Performance

### Peak output throughput @ C=16, baseline (tok/s; % of row best)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 174.1 (43%) | 344.5 (85%) | 406.6 (100%) |
| M2 · openai/gpt-oss-20b | 171.6† | 228.0† | 270.8† |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 88.5 (32%) | 182.4 (65%) | 278.9 (100%) |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 272.2 (39%) | 424.8 (60%) | 706.2 (100%) |

† **Provisional — excluded from rankings and derived stats.** Output-token accounting fell below threshold (expected 50 × 256 = 12800 tokens): the gpt-oss-20b runs counted ~17% of expected output tokens on all three systems (suspected reasoning-token split under the OpenAI chat endpoint). See each run's `report.md → Data quality`; values must not be ranked until diagnosed (2026-09-08 review, P0).

### Batch scaling, C=1 → C=16 (baseline throughput ratio)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 9.4× | 10.1× | 9.3× |
| M2 · openai/gpt-oss-20b | n/a | n/a | n/a |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 3.1× | 6.8× | 7.3× |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 18.3× | 9.8× | 6.1× |

### kv-fp8 vs baseline @ C=16 (output throughput %)

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | -3.0% | -0.6% | +1.3% |
| M2 · openai/gpt-oss-20b | n/a | n/a | n/a |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | +3.5% | -0.9% | +0.8% |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | -7.2% | -0.4% | +0.1% |

### Takeaways

- The cross-system gap is **narrowest on M1** (dense ~9B, BF16: last place still at 43% of best) and **widest on M3** (dense 27B, AWQ-4bit: 32%).
- **AMD 0x7551 barely scales on M3**: 3.1× from C=1 to C=16 (28 → 88 tok/s) while TPOT p50 climbs 34 → 157 ms — batch decode degrades under load (KV/scheduling pressure on the tight 32 GB fit and/or a ROCm batching inefficiency for this checkpoint).
- **AMD 0x7551 M4 single-stream anomaly**: 14.8 tok/s at C=1 vs 146.8 at C=4 (10× step) — single-stream MoE decode is inefficient on this stack; the 18.3× C=1→C=16 'scaling' partly reflects this, not superlinear batching.
- kv-fp8 hurts **AMD 0x7551** most (mean -2.2%), worst cell M4 -7.2% @ C=16. Single-stream is hit even harder: M3 on AMD 0x7551 -49% @ C=1. No system benefits at this workload size.
- **long-context (32k max-model-len) is a no-op for M1** throughput (only model with that cell): Intel Arc Pro B70 +0.1%, NVIDIA L40 -0.5% vs baseline @ C=16 — expected, since the workload still sends 512-token prompts and only the supported context window grows (AMD's M1 long-context run failed at engine startup; see caveats).

## Latency (ms)

Averages over all baseline cells (4 models × C=1/4/8/16). **p50** = typical request, **p99** = worst 1% of requests. TTFT = time to first token (prefill + queueing); TPOT = per-token decode latency; ITL = inter-token gap (streaming tail risk).

| Metric | AMD 0x7551 p50 | AMD 0x7551 p99 | Intel Arc Pro B70 p50 | Intel Arc Pro B70 p99 | NVIDIA L40 p50 | NVIDIA L40 p99 |
|---|---|---|---|---|---|---|
| TTFT | 858 | 1456 | 790 | 1252 | 649 | 984 |
| TPOT | 68 | 72 | 34 | 37 | 24 | 25 |
| ITL | 65 | 167 | 33 | 74 | 22 | 63 |

### Takeaways

- **TTFT**: single-stream prefill is fast everywhere (C=1 p50 up to 465 ms); under full load (C=16) NVIDIA L40 queues fastest (1135 ms mean p50 across models) vs 1554 ms for Intel Arc Pro B70 — the heavy M3 dense-27B prefill dominates (2.3–3.4 s p50 per request at C=16).
- **TPOT**: NVIDIA L40 decodes fastest at every concurrency level (24 ms mean p50 vs 68 ms for AMD 0x7551). M1 @ C=16 spans 2.5× across systems (29 → 74 ms).
- **ITL tails**: the worst single cell is AMD 0x7551 on cyankiwi/Qwen3.8-27B-AWQ-INT4 @ C=16 (1172 ms p99) — streaming stalls of ~1.2 s. In absolute terms NVIDIA L40 still keeps the worst 1% of inter-token gaps lowest (63 ms vs 167 ms for AMD 0x7551); relative tail width (mean p99/p50) is tightest on Intel Arc Pro B70 (1.6×).

## Power efficiency (output tok/s per watt)

C=16, baseline. AMD/NVIDIA: measured `power_avg_w` over the bench run. Intel: no GPU telemetry captured → evaluated at the documented **230 W max TDP**, so its values are conservative floors.

| Model | AMD 0x7551 | Intel Arc Pro B70 | NVIDIA L40 |
|---|---|---|---|
| M1 · Qwen/Qwen3.5-9B | 0.83 (210 W) | 1.50 (230 W*) | 1.72 (237 W) |
| M2 · openai/gpt-oss-20b | n/a | n/a | n/a |
| M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4 | 0.31 (289 W) | 0.79 (230 W*) | 1.08 (257 W) |
| M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | 1.04 (263 W) | 1.85 (230 W*) | 3.63 (194 W) |
| **Overall (mean)** | **0.72** | **1.38** | **2.14** |

Note: Intel power assumed (max TDP), not measured.

### Takeaways

- **NVIDIA L40 is 3.0× AMD 0x7551 overall.** The gap is largest on M4 (MoE 35B): 3.63 vs 1.04 tok/s/W — strong MoE execution at a moderate 194 W — and smallest on M1: 1.72 vs 0.83.
- Worst efficiency cell: AMD 0x7551 on cyankiwi/Qwen3.8-27B-AWQ-INT4 (0.31 tok/s/W at 289 W) — dense 27B decode is power-hungry on this stack.
- Intel Arc Pro B70's true efficiency is at or above the values shown (assumed full TDP; actual draw was likely lower).

## Caveats

- **Protocol defects (2026-09-08 review, P0)**: these runs used vLLM 0.28.0 defaults with **prefix caching effectively ON in every cell** (verified from server logs — the config only *commented* it off) and a single measured pass per level (2 warmups). With a fixed-seed workload, prompts are replayed at each concurrency level, so prefix-cache hits (up to 76% on M2/AMD @ C=16) inflate throughput — most at high C. The corrected reference (cache off, per-level warmup until no JIT, ≥ 3 passes with server restarts) is implemented in `container/run_matrix.py` + `config/models.yaml`; a re-run (T0) is required before re-ranking. See `docs/Benchmark_GPU_Conclusions_et_plan_de_tests_Benjamin.pdf`.
- **M2 (gpt-oss-20b) is provisional on all systems**: only ~15–20% of the expected 12 800 output tokens were counted († in the tables). Suspected reasoning-token split under the OpenAI chat endpoint (server logs show `reasoning_parser='openai_gptoss'`); unconfirmed. M2 cells are excluded from all derived stats above. Raw API diagnostics are captured automatically on the next run (P0-2).
- **VRAM differs**: NVIDIA L40 has 45 GB vs 32 GB on AMD/Intel. All four models fit comfortably on 32 GB at this workload (~12.3 k KV tokens at C=16), so the extra headroom does not change scheduling; it only matters for long-context cells.
- **AMD**: M1 `long-context` cell failed at engine startup (`failed: engine-startup`); shown as n/a. All other cells ran.
- **Intel**: the XPU run captured **no GPU telemetry** (mem/util/power). Power-efficiency figures for Intel assume the documented **max TDP of 230 W** for the Arc Pro B70 (Intel Arc Pro B-Series spec sheet) — an upper bound on actual draw, so Intel's tok/s-per-W values are conservative floors (true efficiency is at or better than shown).
- **Host CPUs differ**: AMD/Intel runs on an AMD Ryzen 7 9800X3D (consumer), NVIDIA on an Intel Xeon (Sapphire Rapids). Negligible for GPU-bound decode; noted for completeness. The AMD run used GPU index 1 on a dual-GPU host.