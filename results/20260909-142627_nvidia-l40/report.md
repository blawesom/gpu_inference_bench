# GPU Inference Bench Report

**NVIDIA L40** · Run: 20260909-142627_nvidia-l40 · vLLM 0.28.0 · 2026-09-10T06:04:02.199621+00:00

| Metric | Value |
|---|---|
| Total cells | 10 |
| Bench rows | 36 |
| Skipped | 1 |
| Failed | 0 |
| Input tokens | 512 |
| Output tokens | 256 |
| Prompts | 50 |
| Concurrency levels | [1, 4, 8, 16] |
| Measured passes | 3 (server restart between passes; metrics are the **median**, spread in report.json `output_throughput_min/max`) |
| Warmup | per-level, up to 3 extra passes until no new JIT compilations (8 prompts each) |

## Environment

| Field | Value |
|---|---|
| Vendor | nvidia |
| GPU | NVIDIA L40 |
| GPU index (in container) | 0 |
| VRAM (GB) | 45.0 |
| Driver | 610.57.04 |
| Stack | cuda 13.0 |
| OS | Rocky Linux 10.2 (Red Quartz) |
| Kernel | 6.12.0-211.50.1.el10_2.x86_64 |
| CPU | Intel Xeon Processor (SapphireRapids) |
| CPU cores | 16 |
| RAM (GB) | 46.7 |
| GPU kernel modules | nvidia, nvidia_drm, nvidia_modeset, nvidia_uvm |
| Docker | 29.8.0 |
| Image | vllm/vllm-openai:v0.28.0 |
| Image ID | sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14 |
| vLLM | 0.28.0 |
| Telemetry source | n/a |
| Telemetry probe | n/a |

## Model Summary (Concurrency = 1)

| Model | Config | Done | Fail | Req/s | Tok/s | TTFT p99 | TPOT p99 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B | baseline | 50 | 0 | 0.17 | 43.67 | 109.45 | 22.59 |
| Qwen/Qwen3.5-9B | kv-fp8 | 50 | 0 | 0.17 | 43.90 | 109.98 | 22.48 |
| Qwen/Qwen3.5-9B | long-context | 50 | 0 | 0.17 | 43.75 | 105.29 | 22.58 |
| openai/gpt-oss-20b | baseline | 50 | 0 | 1.06 | 45.62 | 55.69 | 47.87 |
| openai/gpt-oss-20b | kv-fp8 | 50 | 0 | 1.09 | 47.30 | 55.41 | 41.31 |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | baseline | 50 | 0 | 0.15 | 37.98 | 285.78 | 25.41 |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | kv-fp8 | 50 | 0 | 0.15 | 38.09 | 289.19 | 25.32 |
| cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | baseline | 50 | 0 | 0.45 | 115.66 | 113.70 | 8.24 |
| cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | kv-fp8 | 50 | 0 | 0.45 | 115.99 | 115.30 | 8.21 |

## M1 · Qwen/Qwen3.5-9B

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aiter-attn | n/a | skipped: aiter-amd-only | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | — | n/a | n/a | n/a | n/a |
| baseline | 1 | ok | 0.17 | 43.67 | 103.82 | 109.45 | 22.58 | 22.59 | 22.56 | 22.94 | 293.08 | — | 38.56 | 99.90 | 286.20 | 6529.40 |
| baseline | 4 | ok | 0.57 | 146.33 | 353.10 | 369.55 | 25.07 | 26.05 | 25.07 | 25.47 | 87.48 | — | 38.68 | 99.80 | 273.70 | 1860 |
| baseline | 8 | ok | 1.00 | 255.37 | 448.88 | 725.24 | 26.15 | 28.00 | 25.68 | 26.92 | 50.12 | — | 38.78 | 98 | 279.60 | 1096.30 |
| baseline | 16 | ok | 1.59 | 406.20 | 888.35 | 1457.34 | 29.15 | 32.19 | 27.25 | 51.24 | 31.51 | — | 38.78 | 100 | 284.40 | 690 |
| kv-fp8 | 1 | ok | 0.17 | 43.90 | 103.49 | 109.98 | 22.47 | 22.48 | 22.46 | 22.77 | 291.60 | — | 38.64 | 99.80 | 294.20 | 6689.30 |
| kv-fp8 | 4 | ok | 0.57 | 146.82 | 352.90 | 370.79 | 24.98 | 25.95 | 24.97 | 25.30 | 87.18 | — | 38.80 | 99.90 | 273.90 | 1840.40 |
| kv-fp8 | 8 | ok | 1.00 | 256.60 | 450.07 | 730.54 | 26.01 | 27.86 | 25.55 | 26.77 | 49.88 | — | 38.86 | 100 | 285.80 | 1094.90 |
| kv-fp8 | 16 | ok | 1.59 | 407.36 | 906.75 | 1453.12 | 29.01 | 32.08 | 27.02 | 52.55 | 31.42 | — | 38.86 | 96.90 | 277.40 | 675.10 |
| long-context | 1 | ok | 0.17 | 43.75 | 96.92 | 105.29 | 22.57 | 22.58 | 22.56 | 22.86 | 292.60 | — | 38.56 | 99.80 | 282.90 | 6454 |
| long-context | 4 | ok | 0.57 | 146.36 | 351.85 | 359.81 | 25.08 | 26.02 | 25.07 | 25.42 | 87.45 | — | 38.68 | 99.70 | 262 | 1781.70 |
| long-context | 8 | ok | 1.00 | 255.13 | 441.21 | 722.90 | 26.17 | 28.08 | 25.70 | 26.90 | 50.17 | — | 38.78 | 100 | 274.20 | 1050.30 |
| long-context | 16 | ok | 1.58 | 404.64 | 899.65 | 1465.12 | 29.27 | 32.32 | 27.27 | 50.82 | 31.63 | — | 38.78 | 100 | 291.20 | 706.60 |

## M2 · openai/gpt-oss-20b

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 1.06 | 45.62 | 54.62 | 55.69 | 24.00 | 47.87 | 6.36 | 19.00 | 47.25 | output-token-shortfall | 40.56 | 100 | 283.30 | n/a |
| baseline | 4 | ok | 2.59 | 110.03 | 62.34 | 157.34 | 39.50 | 79.41 | 9.78 | 44.49 | 19.31 | output-token-shortfall | 40.60 | 100 | 274.50 | n/a |
| baseline | 8 | ok | 4.06 | 168.15 | 66.98 | 300.03 | 48.58 | 97.20 | 11.77 | 48.94 | 12.31 | output-token-shortfall | 40.60 | 100 | 275 | n/a |
| baseline | 16 | ok | 5.74 | 225.90 | 103.40 | 1212.81 | 61.65 | 115.69 | 13.66 | 84.05 | 8.71 | output-token-shortfall | 40.60 | 100 | 270.30 | n/a |
| kv-fp8 | 1 | ok | 1.09 | 47.30 | 53.90 | 55.41 | 24.77 | 41.31 | 6.34 | 18.75 | 46.08 | output-token-shortfall | 40.56 | 100 | 278.40 | n/a |
| kv-fp8 | 4 | ok | 2.64 | 108.99 | 63.58 | 168.80 | 45.14 | 73.01 | 9.75 | 46.56 | 18.92 | output-token-shortfall | 40.60 | 100 | 288 | n/a |
| kv-fp8 | 8 | ok | 4.26 | 165.87 | 66.78 | 314.19 | 51.06 | 92.26 | 11.65 | 52.58 | 11.73 | output-token-shortfall | 40.60 | 100 | 296.80 | n/a |
| kv-fp8 | 16 | ok | 5.76 | 240.63 | 101.18 | 612.01 | 60.52 | 132.33 | 13.70 | 81.64 | 8.67 | output-token-shortfall | 40.60 | 100 | 293.10 | n/a |

## M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.15 | 37.98 | 258.98 | 285.78 | 25.40 | 25.41 | 25.40 | 25.65 | 337.01 | — | 40.83 | 99.70 | 293.40 | 7703.20 |
| baseline | 4 | ok | 0.47 | 119.06 | 1042.65 | 1119.00 | 28.48 | 29.80 | 28.47 | 28.78 | 107.51 | — | 41.02 | 99.90 | 276.20 | 2288 |
| baseline | 8 | ok | 0.77 | 196.40 | 1573.17 | 2071.05 | 31.18 | 34.74 | 29.45 | 30.01 | 65.17 | — | 41.10 | 98.50 | 290.20 | 1476.80 |
| baseline | 16 | ok | 1.09 | 278.65 | 2528.03 | 4241.05 | 39.91 | 47.10 | 33.55 | 179.86 | 45.94 | — | 41.10 | 100 | 297.90 | 1048.60 |
| kv-fp8 | 1 | ok | 0.15 | 38.09 | 266.55 | 289.19 | 25.30 | 25.32 | 25.30 | 25.53 | 336.03 | — | 40.91 | 99.70 | 294.70 | 7714.60 |
| kv-fp8 | 4 | ok | 0.47 | 119.94 | 1050.32 | 1088.82 | 28.28 | 29.58 | 28.27 | 28.57 | 106.72 | — | 41.15 | 99.90 | 283.50 | 2326.10 |
| kv-fp8 | 8 | ok | 0.77 | 197.74 | 1580.35 | 2076.88 | 30.94 | 34.46 | 29.18 | 29.95 | 64.73 | — | 41.20 | 98.50 | 294.80 | 1500.10 |
| kv-fp8 | 16 | ok | 1.11 | 283.41 | 2497.83 | 4131.10 | 39.19 | 46.26 | 33.02 | 176.44 | 45.16 | — | 41.22 | 100 | 294.80 | 1015.70 |

## M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.45 | 115.66 | 112.31 | 113.70 | 8.24 | 8.24 | 8.24 | 8.36 | 110.66 | — | 40.89 | 96.80 | 269.10 | 2313.60 |
| baseline | 4 | ok | 1.25 | 320.87 | 218.77 | 228.90 | 11.28 | 11.39 | 11.35 | 11.78 | 39.89 | — | 40.98 | 98.40 | 235.70 | 718.80 |
| baseline | 8 | ok | 1.89 | 483.83 | 334.36 | 370.82 | 14.23 | 14.71 | 14.02 | 14.76 | 26.46 | — | 41 | 94 | 243.60 | 477.50 |
| baseline | 16 | ok | 2.76 | 706.08 | 349.08 | 666.22 | 18.67 | 19.15 | 17.92 | 111.93 | 18.13 | — | 41 | 92.10 | 254.70 | 341.50 |
| kv-fp8 | 1 | ok | 0.45 | 115.99 | 113.47 | 115.30 | 8.21 | 8.21 | 8.21 | 8.34 | 110.35 | — | 41.01 | 96.60 | 267.30 | 2297.50 |
| kv-fp8 | 4 | ok | 1.26 | 321.74 | 221.58 | 232.42 | 11.21 | 11.32 | 11.32 | 11.69 | 39.78 | — | 41.10 | 95.10 | 247.10 | 755.40 |
| kv-fp8 | 8 | ok | 1.90 | 485.69 | 342.49 | 375.69 | 14.14 | 14.60 | 13.92 | 14.68 | 26.35 | — | 41.12 | 93.80 | 254.50 | 498.80 |
| kv-fp8 | 16 | ok | 2.79 | 713.59 | 352.83 | 673.06 | 18.44 | 18.92 | 17.66 | 114.07 | 17.94 | — | 41.12 | 92.80 | 266.30 | 357.90 |

## Data quality

| Cell | Issue | Detail |
|---|---|---|
| M1 / aiter-attn | **prefix-caching-unverified** | could not parse enable_prefix_caching from server log |
| openai/gpt-oss-20b / baseline C=1 | **output-token-shortfall** | output tokens 2104/12800 (16%) |
| openai/gpt-oss-20b / baseline C=4 | **output-token-shortfall** | output tokens 2106/12800 (16%) |
| openai/gpt-oss-20b / baseline C=8 | **output-token-shortfall** | output tokens 2021/12800 (16%) |
| openai/gpt-oss-20b / baseline C=16 | **output-token-shortfall** | output tokens 2063/12800 (16%) |
| openai/gpt-oss-20b / kv-fp8 C=1 | **output-token-shortfall** | output tokens 2661/12800 (21%) |
| openai/gpt-oss-20b / kv-fp8 C=4 | **output-token-shortfall** | output tokens 2397/12800 (19%) |
| openai/gpt-oss-20b / kv-fp8 C=8 | **output-token-shortfall** | output tokens 2186/12800 (17%) |
| openai/gpt-oss-20b / kv-fp8 C=16 | **output-token-shortfall** | output tokens 1931/12800 (15%) |
| run | **power-window-aligned** | power/energy aligned to the measured bench window (last `duration` s of client wall time); energy = GPU-only (vendor power sensor), not system energy |

(prefix caching verified OFF in 9 cell(s) — not listed above)
