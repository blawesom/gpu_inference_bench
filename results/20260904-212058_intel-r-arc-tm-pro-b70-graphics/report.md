# GPU Inference Bench Report

**Intel(R) Arc(TM) Pro B70 Graphics** · Run: 20260904-212058_intel-r-arc-tm-pro-b70-graphics · vLLM 0.28.0+xpu · 2026-09-04T21:01:42.214212+00:00

| Metric | Value |
|---|---|
| Total cells | 9 |
| Bench rows | 36 |
| Skipped | 0 |
| Failed | 0 |
| Input tokens | 512 |
| Output tokens | 256 |
| Prompts | 50 |
| Concurrency levels | [1, 4, 8, 16] |

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

## Model Summary (Concurrency = 1)

| Model | Config | Done | Fail | Req/s | Tok/s | TTFT p99 | TPOT p99 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B | baseline | 50 | 0 | 0.13 | 34.17 | 95.12 | 29.02 |
| Qwen/Qwen3.5-9B | kv-fp8 | 50 | 0 | 0.13 | 33.94 | 96.04 | 29.21 |
| Qwen/Qwen3.5-9B | long-context | 50 | 0 | 0.13 | 34.18 | 95.69 | 29.02 |
| openai/gpt-oss-20b | baseline | 50 | 0 | 0.75 | 29.80 | 86.86 | 62.62 |
| openai/gpt-oss-20b | kv-fp8 | 50 | 0 | 0.83 | 29.84 | 88.01 | 55.98 |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | baseline | 50 | 0 | 0.11 | 26.98 | 378.56 | 35.74 |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | kv-fp8 | 50 | 0 | 0.10 | 26.65 | 381.41 | 36.20 |
| cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | baseline | 50 | 0 | 0.17 | 43.15 | 101.58 | 23.05 |
| cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | kv-fp8 | 50 | 0 | 0.17 | 42.44 | 102.72 | 23.39 |

## M1 · Qwen/Qwen3.5-9B

| Config | C | Status | Req/s | Tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Mem peak GB | Util % | Power W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.13 | 34.17 | 94.38 | 95.12 | 29.01 | 29.02 | 28.99 | 29.23 | 374.63 | n/a | n/a | n/a |
| baseline | 4 | ok | 0.48 | 123.74 | 304.49 | 310.57 | 30.10 | 30.83 | 30.04 | 30.74 | 103.44 | n/a | n/a | n/a |
| baseline | 8 | ok | 0.84 | 215.90 | 333.34 | 592.84 | 31.67 | 33.14 | 31.28 | 35.36 | 59.29 | n/a | n/a | n/a |
| baseline | 16 | ok | 1.35 | 344.50 | 685.85 | 1253.07 | 35.81 | 37.87 | 33.87 | 94.98 | 37.16 | n/a | n/a | n/a |
| kv-fp8 | 1 | ok | 0.13 | 33.94 | 95.45 | 96.04 | 29.20 | 29.21 | 29.20 | 29.40 | 377.12 | n/a | n/a | n/a |
| kv-fp8 | 4 | ok | 0.48 | 123.01 | 307.72 | 314.82 | 30.27 | 31.00 | 30.25 | 30.76 | 104.05 | n/a | n/a | n/a |
| kv-fp8 | 8 | ok | 0.84 | 215.36 | 336.54 | 598.72 | 31.69 | 33.19 | 31.39 | 31.99 | 59.44 | n/a | n/a | n/a |
| kv-fp8 | 16 | ok | 1.34 | 342.26 | 691.95 | 1265.97 | 36.05 | 38.13 | 34.17 | 95.63 | 37.40 | n/a | n/a | n/a |
| long-context | 1 | ok | 0.13 | 34.18 | 94.40 | 95.69 | 29.00 | 29.02 | 28.98 | 29.22 | 374.52 | n/a | n/a | n/a |
| long-context | 4 | ok | 0.48 | 123.74 | 304.53 | 308.29 | 30.09 | 30.82 | 30.04 | 30.70 | 103.44 | n/a | n/a | n/a |
| long-context | 8 | ok | 0.84 | 215.85 | 332.40 | 592.64 | 31.66 | 33.15 | 31.30 | 35.33 | 59.30 | n/a | n/a | n/a |
| long-context | 16 | ok | 1.35 | 344.68 | 684.20 | 1252.90 | 35.80 | 37.85 | 33.84 | 95.03 | 37.14 | n/a | n/a | n/a |

## M2 · openai/gpt-oss-20b

| Config | C | Status | Req/s | Tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Mem peak GB | Util % | Power W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.75 | 29.80 | 85.83 | 86.86 | 32.18 | 62.62 | 9.24 | 28.32 | 66.42 | n/a | n/a | n/a |
| baseline | 4 | ok | 2.10 | 87.69 | 43.39 | 45.69 | 49.10 | 100.68 | 13.15 | 40.91 | 23.81 | n/a | n/a | n/a |
| baseline | 8 | ok | 3.45 | 155.52 | 50.22 | 53.77 | 57.18 | 121.65 | 15.79 | 48.88 | 14.49 | n/a | n/a | n/a |
| baseline | 16 | ok | 5.38 | 228.03 | 58.82 | 61.32 | 65.21 | 142.67 | 18.87 | 57.49 | 9.30 | n/a | n/a | n/a |
| kv-fp8 | 1 | ok | 0.83 | 29.84 | 87.25 | 88.01 | 31.12 | 55.98 | 9.38 | 28.87 | 60.18 | n/a | n/a | n/a |
| kv-fp8 | 4 | ok | 2.31 | 84.49 | 43.87 | 46.16 | 46.88 | 76.74 | 13.23 | 42.04 | 21.67 | n/a | n/a | n/a |
| kv-fp8 | 8 | ok | 3.63 | 135.01 | 50.49 | 53.19 | 56.10 | 91.79 | 15.75 | 49.60 | 13.78 | n/a | n/a | n/a |
| kv-fp8 | 16 | ok | 5.48 | 212.52 | 58.47 | 61.71 | 64.40 | 108.77 | 18.73 | 58.01 | 9.13 | n/a | n/a | n/a |

## M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4

| Config | C | Status | Req/s | Tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Mem peak GB | Util % | Power W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.11 | 26.98 | 377.70 | 378.56 | 35.73 | 35.74 | 35.70 | 36.78 | 474.42 | n/a | n/a | n/a |
| baseline | 4 | ok | 0.34 | 87.66 | 1500.83 | 1503.62 | 38.51 | 42.78 | 38.47 | 40.08 | 146.03 | n/a | n/a | n/a |
| baseline | 8 | ok | 0.54 | 139.24 | 1539.76 | 2871.37 | 43.10 | 51.68 | 41.60 | 43.18 | 91.93 | n/a | n/a | n/a |
| baseline | 16 | ok | 0.71 | 182.37 | 3411.47 | 6064.63 | 65.08 | 76.52 | 55.16 | 408.31 | 70.19 | n/a | n/a | n/a |
| kv-fp8 | 1 | ok | 0.10 | 26.65 | 380.18 | 381.41 | 36.19 | 36.20 | 36.16 | 37.23 | 480.39 | n/a | n/a | n/a |
| kv-fp8 | 4 | ok | 0.34 | 86.80 | 1509.35 | 1511.03 | 38.91 | 43.21 | 38.87 | 40.53 | 147.46 | n/a | n/a | n/a |
| kv-fp8 | 8 | ok | 0.54 | 138.38 | 1548.41 | 2883.38 | 43.35 | 51.96 | 41.85 | 43.59 | 92.50 | n/a | n/a | n/a |
| kv-fp8 | 16 | ok | 0.71 | 180.64 | 3429.50 | 6102.65 | 65.75 | 77.25 | 55.77 | 410.19 | 70.86 | n/a | n/a | n/a |

## M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit

| Config | C | Status | Req/s | Tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Mem peak GB | Util % | Power W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.17 | 43.15 | 99.37 | 101.58 | 22.86 | 23.05 | 22.96 | 23.45 | 296.63 | n/a | n/a | n/a |
| baseline | 4 | ok | 0.63 | 161.15 | 282.45 | 285.48 | 22.92 | 23.71 | 22.98 | 23.47 | 79.43 | n/a | n/a | n/a |
| baseline | 8 | ok | 1.13 | 288.06 | 288.25 | 523.99 | 23.58 | 24.79 | 23.05 | 24.66 | 44.44 | n/a | n/a | n/a |
| baseline | 16 | ok | 1.66 | 424.83 | 565.42 | 1043.56 | 29.47 | 31.06 | 28.23 | 103.24 | 30.13 | n/a | n/a | n/a |
| kv-fp8 | 1 | ok | 0.17 | 42.44 | 100.70 | 102.72 | 23.27 | 23.39 | 23.34 | 23.80 | 301.62 | n/a | n/a | n/a |
| kv-fp8 | 4 | ok | 0.62 | 158.06 | 286.84 | 291.01 | 23.37 | 24.14 | 23.42 | 24.02 | 80.98 | n/a | n/a | n/a |
| kv-fp8 | 8 | ok | 1.11 | 283.47 | 332.87 | 532.64 | 23.99 | 25.21 | 23.45 | 24.56 | 45.15 | n/a | n/a | n/a |
| kv-fp8 | 16 | ok | 1.65 | 422.93 | 593.91 | 1002.99 | 29.25 | 31.17 | 28.31 | 69.88 | 30.27 | n/a | n/a | n/a |
