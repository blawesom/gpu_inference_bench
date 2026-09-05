# Budget / Performance-per-Dollar Recommendation

Three-card fleet evaluation for `gpu_inference_bench`: **NVIDIA L40**, **AMD Radeon AI PRO R9700**, **Intel Arc Pro B70** — measured vLLM throughput (see `results/` runs) combined with US retail prices (fetched 2026-09-05/06).

> Card label note: the AMD run (`results/20260903-222650_0x7551`) used the **Radeon AI PRO R9700** (PCI device 0x7551). Earlier docs/comments referred to it as "RX 9070 XT 32GB" — same silicon family (Navi 32, 32 GB GDDR6), different SKU; this doc uses R9700 throughout.

## TL;DR verdict

| Card | 32/48 GB | Price (new) | Verdict |
|---|---|---|---|
| **Intel Arc Pro B70** | 32 | **$1,300** (MSRP $949) | **Best budget/performance** — 73% of L40 aggregate throughput at ~1/5 the price; $4.66 per 100 tok/s |
| **NVIDIA L40** | 48 | **~$6,500 street (est.)** (MSRP $8,999) | **Performance + power-efficiency champion** — 384 tok/s geomean, 1.97 tok/s/W, most mature stack; worst $/tok per card |
| **AMD Radeon AI PRO R9700** | 32 | **$1,700** (MSRP $1,299) | **Avoid for pure inference today** — only 43% of L40 at 130% of B70's price; ROCm/vLLM maturity issues (M3 batch collapse, kv-fp8 regressions, M1 long-context startup failure) |

---

## 1. Input data

### 1.1 Benchmark data (identical workload, vLLM v0.28.0)

Workload: random dataset, 512-in / 256-out tokens, 50 prompts/level, 2 warmups, seed 42, temp 0, `--ignore-eos`, concurrency 1/4/8/16. Model matrix: M1 `Qwen3.5-9B` (BF16) · M2 `gpt-oss-20b` (MXFP4) · M3 `Qwen3.8-27B-AWQ-INT4` · M4 `Qwen3.5-35B-A3B-AWQ-4bit`.

| Card | Run dir | Stack | VRAM (measured) |
|---|---|---|---|
| AMD Radeon AI PRO R9700 | `results/20260903-222650_0x7551` | ROCm 7.2.53211, vLLM 0.28.0+rocm723 | 31.9 GB |
| Intel Arc Pro B70 | `results/20260904-212058_intel-r-arc-tm-pro-b70-graphics` | XPU, vLLM 0.28.0+xpu | 31.9 GB |
| NVIDIA L40 | `results/20260905-134350_nvidia-l40` | CUDA 13.0, vLLM 0.28.0 | 45.0 GB (48 GB) |

### 1.2 Retail prices (US, fetched 2026-09-05/06)

| Card | MSRP | Current lowest (new) | 3-mo median | 12-mo low | Source |
|---|---|---|---|---|---|
| Intel Arc Pro B70 | $949 | **$1,300** | $1,043 | $999 | gpuprix.com/us/gpus/arc-pro-b70 (Newegg ASRock B70 CT $1,300 open-box; AMZ ASRock Creator $1,300; Walmart/MicroCenter listings) |
| AMD Radeon AI PRO R9700 | $1,299 | **$1,700** | $1,460 | $1,250 | gpuprix.com/us/gpus/radeon-ai-pro-r9700 (AMZ ASRock Creator $1,700; Newegg open-box $1,188 out of stock) |
| NVIDIA L40 | $8,999 (launch 2023) | not listed on consumer trackers (datacenter SKU: CDW/Insight/OEM channel; EOL, superseded by L40S/RTX PRO 5000) | — | — | **~$6,500 street = working estimate for this doc (not verified against a live listing)** |

Market context: all three cards trade above MSRP in the 2026 memory-supply squeeze (B70 +37% vs MSRP, R9700 +31%, per gpuprix). The B70 is 25% above its own 3-month median — pricing is volatile and trending up; a wait may improve value (at the $1,043 median the B70's $/tok would rise ~19%).

---

## 2. Measured performance recap

### 2.1 Throughput, C=16 baseline (output tok/s; best in row bold)

| Model | L40 | Arc Pro B70 | R9700 |
|---|---|---|---|
| M1 9B dense | **406.6** | 344.5 (85%) | 174.1 (43%) |
| M2 20B MoE | **270.8** | 228.0 (84%) | 171.6 (63%) |
| M3 27B dense | **278.9** | 182.4 (65%) | 88.5 (32%) |
| M4 35B MoE | **706.2** | 424.8 (60%) | 272.2 (39%) |
| **Geomean** | **383.8** | **279.0 (72.7%)** | **163.8 (42.7%)** |

### 2.2 Single-user latency (TPOT p50 @ C=1, ms; lower is better)

| Model | R9700 | B70 | L40 |
|---|---|---|---|
| M1 | 53.8 | 29.0 | **22.6** |
| M2 | 72.1 | 32.2 | **23.1** |
| M3 | **33.9** | 35.7 | 25.4 |
| M4 | 71.7 (C=1 anomaly) | 22.9 | **8.3** |

### 2.3 Batching scaling (baseline, C=1 → C=16)

| Model | R9700 | B70 | L40 |
|---|---|---|---|
| M1 | 9.4× | 10.1× | 9.3× |
| M2 | 10.6× | 7.7× | 6.3× |
| M3 | **3.1×** ⚠ | 6.8× | 7.3× |
| M4 | 18.3× (distorted by C=1 anomaly) | 9.8× | 6.1× |

### 2.4 kv-fp8 delta vs baseline @ C=16 (%)

| Model | R9700 | B70 | L40 |
|---|---|---|---|
| M1 | −3.0 | −0.6 | +1.3 |
| M2 | **−12.9** ⚠ | −6.8 | −4.1 |
| M3 | +3.5 | −0.9 | +0.8 |
| M4 | −7.2 | −0.4 | +0.1 |

### 2.5 Power efficiency (output tok/s per W, mean over all ok cells @ C=16)

| R9700 | B70 | L40 |
|---|---|---|
| 0.73 (measured) | 1.30 (floor — no telemetry, rated 230 W TDP used) | **1.97 (measured, 194 W avg at M4/C16)** |

### 2.6 Known per-card issues

- **R9700 (ROCm 7.2.53211):** M3 dense-27B barely scales (3.1×; TPOT p50 34→157 ms, ITL p99 1.2 s under load) — KV/scheduling pressure on the tight 32 GB fit plus ROCm batching inefficiency for this checkpoint; kv-fp8 regresses (M2 −12.9%, M1 single-stream 18.5→17.7 and M3 28.1→14.4 tok/s); M1 `long-context` cell **failed at engine startup**; M4 single-stream decode inefficient (14.8 tok/s at C=1 vs 146.8 at C=4). Measured draw 262.6 W avg / 301 W peak (M4/C16).
- **B70:** no power/temperature telemetry captured in the XPU run (floor values assumed); 32 GB tight-fit constraints on M3/M4 (0.95 util, 2048 ctx).
- **L40:** no issues — all 13 cells ran; 48 GB gives headroom the 32 GB cards lack (long-context only runs on B70/L40, not R9700).

---

## 3. Performance per dollar

Prices: B70 $1,300 · R9700 $1,700 · L40 $6,500 (street est.). Headline metric: geomean C=16 baseline throughput across the 4 models (steady-state throughput; C=1 excluded because single-stream numbers are stack-dependent, e.g. the R9700 M4 anomaly).

### 3.1 Aggregate value

| Card | Geomean tok/s | Price | tok/s per $1k | $ per 100 tok/s | Index (B70 = 1.0) |
|---|---|---|---|---|---|
| **Arc Pro B70** | 279.0 | $1,300 | **214.6** | **$4.66** | **1.00** |
| R9700 | 163.8 | $1,700 | 96.4 | $10.38 | 0.45 |
| L40 | 383.8 | $6,500 (est.) | 59.0 | $16.94 | 0.27 |

### 3.2 Per-model price for throughput (C=16 baseline, $ per 100 tok/s)

| Model | R9700 | B70 | L40 |
|---|---|---|---|
| M1 9B | $9.76 | **$3.77** | $15.99 |
| M2 20B MoE | $9.91 | **$5.70** | $24.00 |
| M3 27B | $19.21 | **$7.13** | $23.31 |
| M4 35B MoE | $6.25 | **$3.06** | $9.20 |

The B70 is cheapest on **every** model. L40's per-model gap shrinks on MoE (where its active-parameter efficiency helps) but per dollar it never wins.

### 3.3 Fleet math — how many cards to match one L40

Cards needed to reach/exceed the L40's single-card C=16 baseline throughput (price of that many cards):

| Model (L40 tok/s) | B70: units (price) | R9700: units (price) |
|---|---|---|
| M1 (406.6) | 2 (1.18×; **$2,600**) | 3 (2.34×; $5,100) |
| M2 (270.8) | 2 (1.19×; **$2,600**) | 2 (1.58×; $3,400) |
| M3 (278.9) | 2 (1.53×; **$2,600**) | 4 (3.15×; $6,800) |
| M4 (706.2) | 2 (1.66×; **$2,600**) | 3 (2.59×; $5,100) |

**Two B70s at $2,600 beat one L40 at ~$6,500 on aggregate throughput for all four models** — at the cost of 2 slots and a power envelope of up to 2×230 W (vs 300 W rated for the L40, ~194 W measured under this workload). The R9700 needs 3–4 cards ($5,100–6,800) to match, and its M3 throughput per card is the weak link.

### 3.4 Secondary value metrics

| Card | Price per GB (nominal) | Bandwidth | TDP | tok/s per W |
|---|---|---|---|---|
| Arc Pro B70 | $40.6/GB | 608 GB/s | 230 W | 1.30 (floor) |
| R9700 | $53.1/GB | 644.6 GB/s | 300 W | 0.73 |
| L40 | $135.4/GB | 864 GB/s | 300 W | **1.97** |

---

## 4. Recommendations per card

### Intel Arc Pro B70 — best budget/performance (recommended fleet default)

**Buy if:** you want the most local-LLM throughput per dollar, 32 GB-class dev/eval boxes, or you need to match L40 aggregate numbers with 2 B70s for ~40% of the L40 street price.

- 72.7% of L40 geomean throughput for $1,300 vs ~$6,500 → **2.2× the R9700 and 3.6× the L40 in tok/s per dollar**.
- Single-user latency is excellent for the class (23–36 ms TPOT p50 @ C=1 on 9B/20B/MoE-35B workloads).
- Cleanest config matrix of the 3: every supported cell ran (M1 long-context included), kv-fp8 neutral.
- Caveats: 32 GB ceiling (M3 needs the 0.95-util/2048-ctx fit flags; 32k-context workloads are a no-go at the dense-27B size); no power/telemetry support in vLLM's XPU path yet; current street price is +37% over the $949 MSRP — at the 3-month median ($1,043) value improves ~25%, so a short wait may pay.

### NVIDIA L40 — performance and efficiency reference (keep as benchmark baseline)

**Buy if:** maximum single-card throughput and best power efficiency matter more than capex, 24/7 datacenter-style use, or you need the 48 GB headroom (long-context, bigger KV) and the most mature inference stack.

- Best on every raw-performance axis: 383.8 tok/s geomean, 1.97 tok/s/W, 406→706 tok/s across the model matrix, zero failed cells.
- Weakest $/tok per card ($16.93/100 tok/s) and 3.4× the $/GB of the B70 — its budget case only works as an existing fleet asset or when street pricing drops further (EOL card).
- The 45 GB usable also matters: the 32 GB cards run M3/M4 with near-zero KV headroom; the L40 has ~13 GB more for long-context or higher concurrency.
- Caveats: passive-cooling server card (needs the right chassis/airflow), no live consumer retail channel in 2026 — the $6,500 figure used here is an estimate; verify quote before budgeting.

### AMD Radeon AI PRO R9700 — skip for inference today, watch for ROCm maturity

**Buy if:** the box is primarily graphics/3D/mixed-use and AI is secondary, or you have an ecosystem lock-in to ROCm.

- Worst value of the three at $1,700: only 42.7% of L40 geomean, 2.2× worse $/tok than the B70, and 31% above its own $1,299 MSRP.
- The silicon (4096 SP, 644.6 GB/s — the highest bandwidth of the three) is on par with the B70 on paper; the gap is software: M3 dense model collapses under batching (3.1× scaling, 1.2 s ITL p99), kv-fp8 actively hurts (−12.9% on M2), and the M1 long-context cell cannot even start. These are stack-maturity defects (ROCm 7.2.53211 + vLLM 0.28.0), not hardware limits — re-benchmark after ROCm/vLLM upgrades before dismissing the card long-term.
- Its one relative bright spot: M4 MoE-35B at C=16 (272 tok/s, $6.25/100 tok/s) holds up better than M3, and its M3 single-stream latency (33.9 ms) is actually the fastest of the three for that model.

---

## 5. Methodology and caveats

- **Metric choice:** geomean of C=16 *baseline* output throughput over the 4 models. C=16 is the workload's concurrency ceiling (KV-limited on 32 GB). Baseline (no kv-fp8) because FP8 KV is neutral-to-negative on every system here and would double-count a vendor-specific regression (R9700).
- **Price basis:** lowest *new* US listing per card, gpuprix trackers snapshot 2026-09-05; L40 is an estimate (see §1.2) — all L40 value figures scale linearly if the real quote differs.
- **Hosts differ:** R9700/B70 ran on the same consumer host (Ryzen 7 9800X3D); L40 on a Xeon (Sapphire Rapids) rack server. Negligible for GPU-bound decode, noted for completeness.
- **VRAM differs:** L40 48 GB vs 32 GB. All four models fit comfortably on 32 GB at this workload (~12.3 k KV tokens at C=16), so headroom does not change these numbers; it matters for long-context, higher concurrency, or larger models.
- **B70 power is a floor** (230 W TDP ceiling assumed; no telemetry captured) — its true tok/s/W is at or below 1.30, i.e. the B70's efficiency rank may be slightly better than shown, but it does not overtake the L40 even at 230 W.
- **Runs are single-machine snapshots** (one config each), not repeated trials; see `results/cross-system-comparison.md` for the full per-cell tables.

*Generated 2026-09-05 from `results/cross-system-comparison.md` + gpuprix.com price snapshots (accessed 2026-09-05/06).*