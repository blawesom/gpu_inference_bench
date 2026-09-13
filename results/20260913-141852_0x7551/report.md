# GPU Inference Bench Report

**0x7551** · Run: 20260913-141852_0x7551 · vLLM 0.28.0+rocm723 · 2026-09-13T19:51:49.538937+00:00

| Metric | Value |
|---|---|
| Total cells | 10 |
| Bench rows | 32 |
| Skipped | 0 |
| Failed | 2 |
| Input tokens | 512 |
| Output tokens | 256 |
| Prompts | 50 |
| Concurrency levels | [1, 4, 8, 16] |
| Measured passes | 3 (server restart between passes; metrics are the **median**, spread in report.json `output_throughput_min/max`) |
| Warmup | per-level, up to 3 extra passes until no new JIT compilations (8 prompts each) |

## Environment

| Field | Value |
|---|---|
| Vendor | amd |
| GPU | 0x7551 |
| GPU index (in container) | 1 |
| VRAM (GB) | 31.9 |
| Driver | 7.2.4 |
| Stack | rocm 7.2.53211, aiter unknown |
| OS | CachyOS |
| Kernel | 7.2.4-1-cachyos |
| CPU | AMD Ryzen 7 9800X3D 8-Core Processor |
| CPU cores | 16 |
| RAM (GB) | 31.0 |
| GPU kernel modules | amdgpu |
| Docker | 29.8.0 |
| Image | vllm/vllm-openai-rocm:v0.28.0 |
| Image ID | sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032 |
| vLLM | 0.28.0+rocm723 |
| Telemetry source | n/a |
| Telemetry probe | n/a |

## Model Summary (Concurrency = 1)

| Model | Config | Done | Fail | Req/s | Tok/s | TTFT p99 | TPOT p99 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B | baseline | 50 | 0 | 0.07 | 18.95 | 139.63 | 52.54 |
| Qwen/Qwen3.5-9B | kv-fp8 | 50 | 0 | 0.07 | 18.10 | 139.43 | 55.03 |
| openai/gpt-oss-20b | baseline | 50 | 0 | 0.11 | 4.17 | 254.98 | 466.75 |
| openai/gpt-oss-20b | kv-fp8 | 50 | 0 | 0.11 | 4.40 | 258.88 | 433.52 |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | baseline | 50 | 0 | 0.11 | 28.10 | 478.81 | 33.86 |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | kv-fp8 | 50 | 0 | 0.10 | 24.60 | 475.17 | 39.09 |
| cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | baseline | 50 | 0 | 0.07 | 16.69 | 161.14 | 70.52 |
| cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | kv-fp8 | 50 | 0 | 0.23 | 57.87 | 111.53 | 16.98 |

## M1 · Qwen/Qwen3.5-9B

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aiter-attn | n/a | failed: pass1-failed:engine-startup | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | — | n/a | n/a | n/a | n/a |
| baseline | 1 | ok | 0.07 | 18.95 | 138.77 | 139.63 | 52.44 | 52.54 | 52.43 | 52.61 | 675.43 | — | 26.44 | 99.90 | 193.80 | 10205.70 |
| baseline | 4 | ok | 0.49 | 125.96 | 343.94 | 345.01 | 29.44 | 30.29 | 29.44 | 29.65 | 101.62 | — | 26.44 | 100 | 225.90 | 3197 |
| baseline | 8 | ok | 0.67 | 171.14 | 406.11 | 675.36 | 41.80 | 43.42 | 41.55 | 41.89 | 74.79 | — | 26.44 | 100 | 236.50 | 2183.10 |
| baseline | 16 | ok | 1.09 | 277.77 | 757.81 | 1279.65 | 47.46 | 49.81 | 45.92 | 62.99 | 46.08 | — | 26.44 | 100 | 233.20 | 1296.10 |
| kv-fp8 | 1 | ok | 0.07 | 18.10 | 138.63 | 139.43 | 54.94 | 55.03 | 54.94 | 55.57 | 707.34 | — | 26.44 | 100 | 191.30 | 10554.40 |
| kv-fp8 | 4 | ok | 0.46 | 116.65 | 345.71 | 346.54 | 31.88 | 32.72 | 31.87 | 32.50 | 109.73 | — | 26.44 | 100 | 279.80 | 2363 |
| kv-fp8 | 8 | ok | 0.63 | 161.50 | 408.44 | 685.57 | 44.25 | 45.87 | 43.98 | 44.80 | 79.26 | — | 26.44 | 100 | 290.40 | 1773 |
| kv-fp8 | 16 | ok | 1.03 | 262.55 | 773.27 | 1303.85 | 50.09 | 52.46 | 48.44 | 67.50 | 48.75 | — | 26.44 | 99.30 | 279.50 | 1055.70 |
| long-context | n/a | failed: pass1-failed:engine-startup | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | — | n/a | n/a | n/a | n/a |

## M2 · openai/gpt-oss-20b

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.11 | 4.17 | 253.90 | 254.98 | 231.62 | 466.75 | 67.50 | 202.25 | 474.54 | output-token-shortfall | 24.85 | 100 | 169.10 | n/a |
| baseline | 4 | ok | 0.76 | 29.46 | 151.60 | 325.89 | 135.11 | 277.98 | 35.52 | 114.25 | 66.16 | output-token-shortfall | 25.17 | 100 | 292.20 | n/a |
| baseline | 8 | ok | 2.28 | 109.63 | 117.28 | 461.91 | 84.91 | 166.77 | 19.80 | 112.46 | 21.91 | output-token-shortfall | 25.17 | 100 | 295.90 | n/a |
| baseline | 16 | ok | 3.27 | 142.11 | 176.36 | 1159.58 | 101.70 | 202.27 | 22.06 | 133.84 | 15.31 | output-token-shortfall | 25.17 | 99.90 | 295.50 | n/a |
| kv-fp8 | 1 | ok | 0.11 | 4.40 | 258.31 | 258.88 | 228.06 | 433.52 | 67.47 | 202.14 | 436.37 | output-token-shortfall | 24.85 | 100 | 169.40 | n/a |
| kv-fp8 | 4 | ok | 0.79 | 32.47 | 157.27 | 262.72 | 136.29 | 300.92 | 35.67 | 114.79 | 63.10 | output-token-shortfall | 25.17 | 100 | 294.10 | n/a |
| kv-fp8 | 8 | ok | 2.36 | 90.42 | 121.55 | 495.54 | 82.76 | 135.27 | 19.73 | 122.23 | 21.15 | output-token-shortfall | 25.17 | 100 | 296.50 | n/a |
| kv-fp8 | 16 | ok | 3.52 | 151.79 | 176.49 | 1580.96 | 97.81 | 184.24 | 22.38 | 152.45 | 14.22 | output-token-shortfall | 25.17 | 93.10 | 295.70 | n/a |

## M3 · cyankiwi/Qwen3.8-27B-AWQ-INT4

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.11 | 28.10 | 477.64 | 478.81 | 33.86 | 33.86 | 33.85 | 34.15 | 455.53 | — | 27.40 | 100 | 298.60 | 10590.40 |
| baseline | 4 | ok | 0.24 | 60.25 | 1753.02 | 1755.23 | 57.68 | 62.48 | 57.66 | 58.05 | 212.46 | — | 27.72 | 100 | 299.50 | 4961.40 |
| baseline | 8 | ok | 0.22 | 57.15 | 2039.21 | 3501.48 | 124.70 | 134.10 | 123.39 | 124.47 | 223.98 | — | 27.72 | 100 | 299.40 | 5217.10 |
| baseline | 16 | ok | 0.36 | 93.14 | 3611.57 | 6918.39 | 144.49 | 173.46 | 132.13 | 491.46 | 137.43 | — | 27.72 | 100 | 299.40 | 3204.40 |
| kv-fp8 | 1 | ok | 0.10 | 24.60 | 473.67 | 475.17 | 38.97 | 39.09 | 38.94 | 40.07 | 520.36 | — | 27.40 | 100 | 283 | 11477.20 |
| kv-fp8 | 4 | ok | 0.22 | 56.12 | 1708.10 | 1710.46 | 62.55 | 67.21 | 62.45 | 63.72 | 228.07 | — | 27.72 | 100 | 299.60 | 5313 |
| kv-fp8 | 8 | ok | 0.22 | 55.20 | 1988.26 | 3430.50 | 129.31 | 138.43 | 128.09 | 130.01 | 231.88 | — | 27.72 | 100 | 299.60 | 5383.20 |
| kv-fp8 | 16 | ok | 0.36 | 91.44 | 3682.72 | 6844.09 | 146.02 | 159.05 | 136.39 | 477.17 | 139.98 | — | 27.72 | 99.30 | 297.80 | 3253.70 |

## M4 · cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit

| Config | C | Status | Req/s | Tok/s (median) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Flags | Mem peak GB | Util % | Power W | J/ktok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | ok | 0.07 | 16.69 | 155.83 | 161.14 | 70.22 | 70.52 | 70.22 | 70.73 | 766.70 | — | 27.72 | 99.80 | 164 | 9807.30 |
| baseline | 4 | ok | 0.58 | 147.56 | 311.78 | 329.30 | 25.76 | 26.93 | 26.43 | 27.53 | 86.74 | — | 27.72 | 100 | 296.40 | 1992.70 |
| baseline | 8 | ok | 0.73 | 186.45 | 359.03 | 587.34 | 40.29 | 41.71 | 40.36 | 45.33 | 68.65 | — | 27.72 | 98.60 | 292.20 | 1570.90 |
| baseline | 16 | ok | 1.07 | 273.17 | 648.94 | 1089.17 | 52.59 | 54.50 | 52.78 | 82.72 | 46.86 | — | 27.72 | 100 | 293.50 | 1055.60 |
| kv-fp8 | 1 | ok | 0.23 | 57.87 | 107.93 | 111.53 | 16.92 | 16.98 | 16.92 | 17.69 | 221.20 | — | 27.72 | 99.10 | 242.90 | 4175.70 |
| kv-fp8 | 4 | ok | 0.51 | 131.05 | 313.92 | 320.70 | 29.11 | 30.10 | 29.70 | 31.21 | 97.68 | — | 27.72 | 100 | 285.70 | 2166.90 |
| kv-fp8 | 8 | ok | 0.67 | 171.23 | 361.66 | 601.04 | 43.60 | 44.92 | 43.88 | 47.28 | 74.75 | — | 27.72 | 100 | 292.40 | 1693.30 |
| kv-fp8 | 16 | ok | 0.99 | 253.18 | 666.85 | 1114.42 | 56.17 | 58.04 | 56.06 | 88.31 | 50.56 | — | 27.72 | 99.60 | 290.40 | 1139.60 |

## Data quality

| Cell | Issue | Detail |
|---|---|---|
| openai/gpt-oss-20b / baseline C=1 | **output-token-shortfall** | output tokens 1980/12800 (16%) |
| openai/gpt-oss-20b / baseline C=4 | **output-token-shortfall** | output tokens 2466/12800 (19%) |
| openai/gpt-oss-20b / baseline C=8 | **output-token-shortfall** | output tokens 2091/12800 (16%) |
| openai/gpt-oss-20b / baseline C=16 | **output-token-shortfall** | output tokens 2143/12800 (17%) |
| openai/gpt-oss-20b / kv-fp8 C=1 | **output-token-shortfall** | output tokens 1922/12800 (15%) |
| openai/gpt-oss-20b / kv-fp8 C=4 | **output-token-shortfall** | output tokens 2053/12800 (16%) |
| openai/gpt-oss-20b / kv-fp8 C=8 | **output-token-shortfall** | output tokens 1851/12800 (14%) |
| openai/gpt-oss-20b / kv-fp8 C=16 | **output-token-shortfall** | output tokens 2271/12800 (18%) |
| M1 / baseline | **baseline-triton-fallback** | baseline falls back to Triton (expected on this stack; it is the T1 comparison baseline) |
| M3 / baseline | **baseline-triton-fallback** | baseline falls back to Triton (expected on this stack; it is the T1 comparison baseline) |
| M4 / baseline | **baseline-triton-fallback** | baseline falls back to Triton (expected on this stack; it is the T1 comparison baseline) |
| run | **power-window-aligned** | power/energy aligned to the measured bench window (last `duration` s of client wall time); energy = GPU-only (vendor power sensor), not system energy |

(prefix caching verified OFF in 10 cell(s) — not listed above)
