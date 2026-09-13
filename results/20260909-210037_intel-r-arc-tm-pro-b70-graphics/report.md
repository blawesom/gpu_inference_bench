# GPU Inference Bench Report

**Intel(R) Arc(TM) Pro B70 Graphics** · Run: 20260909-210037_intel-r-arc-tm-pro-b70-graphics · vLLM 0.28.0+xpu · 2026-09-10T00:40:22.056275+00:00

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
| Vendor | intel |
| GPU | Intel(R) Arc(TM) Pro B70 Graphics |
| GPU index (in container) | 0 |
| VRAM (GB) | 31.9 |
| Driver | n/a |
| Stack | n/a |
| OS | CachyOS |
| Kernel | 7.2.3-1-cachyos |
| CPU | AMD Ryzen 7 9800X3D 8-Core Processor |
| CPU cores | 16 |
| RAM (GB) | 31.0 |
| GPU kernel modules | amdgpu, xe |
| Docker | 29.7.2 |
| Image | vllm/vllm-openai-xpu:v0.28.0 |
| Image ID | sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3 |
| vLLM | 0.28.0+xpu |
| Telemetry source | n/a |
| Telemetry probe | no-samples |

## Model Summary (Concurrency = 1)

| Model | Config | Done | Fail | Req/s | Tok/s | TTFT p99 | TPOT p99 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B | baseline | 50 | 0 | 0.13 | 34.23 | 94.63 | 28.97 |
| Qwen/Qwen3.5-9B | kv-fp8 | 50 | 0 | 0.13 | 33.98 | 95.80 | 29.18 |
| Qwen/Qwen3.5-9B | long-context | 50 | 0 | 0.13 | 34.23 | 95.14 | 28.97 |
| openai/gpt-oss-20b | baseline | 50 | 0 | 0.72 | 31.31 | 95.95 | 65.82 |
| openai/gpt-oss-20b | kv-fp8 | 50 | 0 | 0.78 | 33.38 | 96.76 | 58.36 |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | baseline | 50 | 0 | 0.11 | 27.00 | 377.99 | 35.72 |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | kv-fp8 | 50 | 0 | 0.10 | 26.67 | 381.05 | 36.17 |
| cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | baseline | 50 | 0 | 0.18 | 46.45 | 101.03 | 21.42 |
| cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | kv-fp8 | 50 | 0 | 0.18 | 45.76 | 103.57 | 21.89 |

## M1 · Qwen/Qwen3.5-9B

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aiter-attn | n/a | skipped: aiter-amd-only | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | — | n/a | n/a | n/a | n/a |
| baseline | 1 | ok | 0.13 | 34.23 | 93.90 | 94.63 | 28.96 | 28.97 | 28.96 | 29.17 | 373.94 | — | 26.59 | n/a | 217.90 | 6389.40 |
| baseline | 4 | ok | 0.48 | 123.99 | 303.92 | 309.59 | 30.04 | 30.76 | 30.02 | 30.50 | 103.23 | — | 26.63 | n/a | 225.80 | 1764.40 |
| baseline | 8 | ok | 0.85 | 216.47 | 370.95 | 634.03 | 31.54 | 33.07 | 31.25 | 31.77 | 59.13 | — | 26.63 | n/a | 228.50 | 974.90 |
| baseline | 16 | ok | 1.35 | 345.81 | 726.24 | 1233.23 | 35.38 | 37.72 | 33.79 | 61.30 | 37.01 | — | 26.63 | n/a | 225.60 | 593.90 |
| kv-fp8 | 1 | ok | 0.13 | 33.98 | 95.08 | 95.80 | 29.18 | 29.18 | 29.17 | 29.36 | 376.74 | — | 26.59 | n/a | 220.50 | 6519.80 |
| kv-fp8 | 4 | ok | 0.48 | 123.18 | 307.12 | 313.32 | 30.23 | 30.96 | 30.21 | 30.70 | 103.91 | — | 26.63 | n/a | 226.90 | 1772.50 |
| kv-fp8 | 8 | ok | 0.84 | 215.54 | 374.12 | 641.02 | 31.64 | 33.20 | 31.34 | 31.91 | 59.38 | — | 26.63 | n/a | 219.10 | 1026.40 |
| kv-fp8 | 16 | ok | 1.34 | 342.51 | 732.12 | 1246.11 | 35.73 | 38.13 | 34.14 | 61.89 | 37.37 | — | 26.63 | n/a | 227 | 591.60 |
| long-context | 1 | ok | 0.13 | 34.23 | 94.01 | 95.14 | 28.96 | 28.97 | 28.96 | 29.15 | 373.96 | — | 26.59 | n/a | 217.70 | 6378.10 |
| long-context | 4 | ok | 0.48 | 124.04 | 303.59 | 309.37 | 30.02 | 30.75 | 30.01 | 30.49 | 103.20 | — | 26.63 | n/a | 225.30 | 1761.80 |
| long-context | 8 | ok | 0.85 | 216.51 | 369.71 | 634.14 | 31.54 | 33.07 | 31.24 | 31.78 | 59.12 | — | 26.63 | n/a | 228.60 | 975.40 |
| long-context | 16 | ok | 1.35 | 345.78 | 726.38 | 1233.28 | 35.37 | 37.75 | 33.80 | 61.34 | 37.02 | — | 26.63 | n/a | 228 | 595 |

## M2 · openai/gpt-oss-20b

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.72 | 31.31 | 94.70 | 95.95 | 32.21 | 65.82 | 9.25 | 28.51 | 69.22 | output-token-shortfall | 28.25 | n/a | 213.20 | n/a |
| baseline | 4 | ok | 1.79 | 72.42 | 104.18 | 251.21 | 54.63 | 107.85 | 13.28 | 77.20 | 27.92 | output-token-shortfall | 28.28 | n/a | 209.60 | n/a |
| baseline | 8 | ok | 2.83 | 130.37 | 110.86 | 487.66 | 66.38 | 126.61 | 15.88 | 108.39 | 17.69 | output-token-shortfall | 28.28 | n/a | 216.40 | n/a |
| baseline | 16 | ok | 3.98 | 157.25 | 180.86 | 1063.06 | 89.68 | 182.36 | 19.19 | 152.86 | 12.57 | output-token-shortfall | 28.28 | n/a | 192.90 | n/a |
| kv-fp8 | 1 | ok | 0.78 | 33.38 | 95.66 | 96.76 | 31.71 | 58.36 | 9.37 | 28.89 | 64.34 | output-token-shortfall | 28.25 | n/a | 219.50 | n/a |
| kv-fp8 | 4 | ok | 1.94 | 78.63 | 105.10 | 279.07 | 51.23 | 99.93 | 13.33 | 78.36 | 25.74 | output-token-shortfall | 28.28 | n/a | 211.40 | n/a |
| kv-fp8 | 8 | ok | 3.00 | 118.04 | 110.92 | 529.16 | 67.23 | 127.01 | 16.04 | 108.27 | 16.69 | output-token-shortfall | 28.28 | n/a | 210.50 | n/a |
| kv-fp8 | 16 | ok | 4.21 | 159.43 | 178.84 | 1070.45 | 84.39 | 153.18 | 18.94 | 156.54 | 11.89 | output-token-shortfall | 28.28 | n/a | 216.40 | n/a |

## M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.11 | 27.00 | 377.20 | 377.99 | 35.71 | 35.72 | 35.68 | 36.76 | 474.12 | — | 28.27 | n/a | 230 | 8426.80 |
| baseline | 4 | ok | 0.34 | 87.77 | 1498.50 | 1500.43 | 38.46 | 42.73 | 38.42 | 40.04 | 145.83 | — | 28.31 | n/a | 226 | 2554.30 |
| baseline | 8 | ok | 0.55 | 139.55 | 1706.60 | 2989.46 | 43.04 | 51.59 | 41.55 | 43.12 | 91.72 | — | 28.31 | n/a | 226 | 1571.20 |
| baseline | 16 | ok | 0.71 | 183.00 | 3295.03 | 5875.29 | 64.46 | 76.26 | 55.08 | 390.97 | 69.95 | — | 28.31 | n/a | 222 | 1195.40 |
| kv-fp8 | 1 | ok | 0.10 | 26.67 | 380.02 | 381.05 | 36.15 | 36.17 | 36.13 | 37.18 | 479.97 | — | 28.29 | n/a | 230 | 8538.10 |
| kv-fp8 | 4 | ok | 0.34 | 86.90 | 1507.50 | 1509.23 | 38.87 | 43.16 | 38.83 | 40.48 | 147.30 | — | 28.33 | n/a | 229.80 | 2553.80 |
| kv-fp8 | 8 | ok | 0.54 | 138.63 | 1715.09 | 3003.52 | 43.32 | 51.91 | 41.81 | 43.53 | 92.33 | — | 28.33 | n/a | 230.10 | 1577.80 |
| kv-fp8 | 16 | ok | 0.71 | 181.46 | 3309.38 | 5903.89 | 65.06 | 76.91 | 55.60 | 392.08 | 70.54 | — | 28.33 | n/a | 229.40 | 1193.30 |

## M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.18 | 46.45 | 99.13 | 101.03 | 21.22 | 21.42 | 21.29 | 21.80 | 275.56 | — | 28.06 | n/a | 156.90 | 3340 |
| baseline | 4 | ok | 0.67 | 171.61 | 282.56 | 285.17 | 21.43 | 22.29 | 21.49 | 21.96 | 74.59 | — | 28.10 | n/a | 183.40 | 1078.80 |
| baseline | 8 | ok | 1.17 | 300.01 | 326.05 | 530.56 | 22.54 | 23.80 | 22.36 | 23.33 | 42.67 | — | 28.10 | n/a | 213.30 | 666 |
| baseline | 16 | ok | 1.69 | 431.69 | 585.31 | 984.73 | 29.09 | 30.95 | 28.18 | 69.06 | 29.65 | — | 28.10 | n/a | 217.80 | 475.10 |
| kv-fp8 | 1 | ok | 0.18 | 45.76 | 100.77 | 103.57 | 21.54 | 21.89 | 21.61 | 22.25 | 279.71 | — | 28.06 | n/a | 156.30 | 3321.10 |
| kv-fp8 | 4 | ok | 0.66 | 169.22 | 286.80 | 290.35 | 21.75 | 22.56 | 21.79 | 22.46 | 75.64 | — | 28.10 | n/a | 185 | 1078.80 |
| kv-fp8 | 8 | ok | 1.16 | 296.03 | 330.75 | 538.83 | 22.95 | 24.18 | 22.64 | 23.64 | 43.24 | — | 28.10 | n/a | 212 | 661.60 |
| kv-fp8 | 16 | ok | 1.67 | 427.38 | 593.95 | 999.94 | 29.20 | 31.07 | 28.27 | 69.75 | 29.95 | — | 28.10 | n/a | 211.60 | 474.90 |

## Data quality

| Cell | Issue | Detail |
|---|---|---|
| M1 / aiter-attn | **prefix-caching-unverified** | could not parse enable_prefix_caching from server log |
| openai/gpt-oss-20b / baseline C=1 | **output-token-shortfall** | output tokens 2167/12800 (17%) |
| openai/gpt-oss-20b / baseline C=4 | **output-token-shortfall** | output tokens 2022/12800 (16%) |
| openai/gpt-oss-20b / baseline C=8 | **output-token-shortfall** | output tokens 2306/12800 (18%) |
| openai/gpt-oss-20b / baseline C=16 | **output-token-shortfall** | output tokens 2148/12800 (17%) |
| openai/gpt-oss-20b / kv-fp8 C=1 | **output-token-shortfall** | output tokens 2148/12800 (17%) |
| openai/gpt-oss-20b / kv-fp8 C=4 | **output-token-shortfall** | output tokens 2067/12800 (16%) |
| openai/gpt-oss-20b / kv-fp8 C=8 | **output-token-shortfall** | output tokens 1973/12800 (15%) |
| openai/gpt-oss-20b / kv-fp8 C=16 | **output-token-shortfall** | output tokens 1911/12800 (15%) |
| run | **power-window-aligned** | power/energy aligned to the measured bench window (last `duration` s of client wall time); energy = GPU-only (vendor power sensor), not system energy |

(prefix caching verified OFF in 9 cell(s) — not listed above)
