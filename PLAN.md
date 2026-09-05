# GPU Inference Benchmark — Platform-Agnostic Benchmark Platform (v2)

> **Deliverable:** A single shell script (`bench.sh`) that detects the local GPU
> (NVIDIA / AMD / Intel), pulls the pinned vLLM image for that vendor, runs the
> inference stack and the benchmark for a 4-model × per-model-optimizations
> matrix, and produces a structured report.

---

## 1. High-level Architecture

```
┌────────────── Host (Linux, x86_64) ─────────────────────────────────┐
│                                                                      │
│  bench.sh                                                            │
│  ├── 1. Preflight: docker, GPU present, disk space, HF cache        │
│  ├── 2. Detect vendor: nvidia | amd | intel                         │
│  ├── 3. Collect host/hardware metadata → environment.json           │
│  ├── 4. docker pull <pinned image for vendor>                       │
│  ├── 5. docker run <vendor device flags>                             │
│  │                                                                      │
│  │   ┌─────────────────── Container ──────────────────────┐          │
│  │   │  entrypoint (run_matrix.py)                        │          │
│  │   │  for model in MODELS:                              │          │
│  │   │    for config in model.configs:                    │          │
│  │   │      vllm serve <model> <config flags> &           │          │
│  │   │      wait healthy → telemetry sampler (1 s) &      │          │
│  │   │      for C in 1 4 8 16:                            │          │
│  │   │        vllm bench serve --max-concurrency C        │          │
│  │   │              --save-result → bench_<...>.json      │          │
│  │   │      kill server                                   │          │
│  │   │  report.py → report.json + report.md               │          │
│  │   └────────────────────────────────────────────────────┘          │
│  └── 6. Results in ./results/<run-id>/                               │
└──────────────────────────────────────────────────────────────────────┘
```

**Benchmark engine = `vllm bench serve`** (built into every vLLM image). We do not
write a custom HTTP client: `vllm bench serve` already measures TTFT / TPOT / ITL
p50/p90/p99, request and token throughput, and supports a random synthetic dataset
with fixed input/output lengths, `--max-concurrency`, and JSON result files.
Our code inside the container is a **thin orchestrator** (start server per
config, run sweep, sample GPU telemetry, aggregate, render report).

---

## 2. Directory Layout

```
gpu_inference_bench/
├── bench.sh                  # THE DELIVERABLE — host entry point
├── config/
│   └── models.yaml           # model matrix + per-model optimization configs
├── container/
│   ├── entrypoint.sh         # pulls env, calls run_matrix.py
│   ├── run_matrix.py         # orchestration: server lifecycle, bench sweeps
│   ├── telemetry.py          # per-vendor 1 Hz GPU sampler
│   └── report.py             # aggregate → report.json + report.md
├── results/                  # output (gitignored); per-run subdirs, .latest pointer
├── PLAN.md
└── README.md
```

No Dockerfiles: vLLM's official pre-built images are the inference environment.
`bench.sh` only does `docker pull`.

---

## 3. Hardware Detection (in `bench.sh`, host-side)

| Vendor  | Detection                                              | `docker run` device flags                                                            |
|---------|--------------------------------------------------------|---------------------------------------------------------------------------------------|
| NVIDIA  | `lspci \| grep -i nvidia` or `nvidia-smi` exits 0     | `--gpus all` (nvidia-container-toolkit required; `--runtime=nvidia` fallback)          |
| AMD     | `lspci \| grep -i 'amd.*vga'` or `/dev/kfd` exists     | `--device=/dev/kfd --device=/dev/dri --group-add=video --security-opt seccomp=unconfined` |
| Intel   | `xpu-smi discovery` succeeds **or** `/sys/module/xe` exists (Arc A/B dGPUs)
|         | **or** `lspci` shows Intel VGA/3D                         | `-v /dev/dri:/dev/dri` (for `by-path/` directory listing) + `--device` per node (for cgroup `open()` permission) — oneCCL does `opendir("/dev/dri/by-path/")` to find `-render` symlinks. Full root-cause: [docs/intel-xpu-dev-dri.md](docs/intel-xpu-dev-dri.md)
|         |                                                        | (`torch.xpu` in-container auto-selects the largest dGPU via VRAM)
|         |                                                        | `--group-add=video --group-add=render` (where groups exist)


No vendor detected → exit with hint. `--vendor` flag can force.
Single-GPU scope: index 0 only, **except when multiple devices are present → pick the one with the largest VRAM**. Set the device index inside the container (`CUDA_VISIBLE_DEVICES=$IDX` for NVIDIA, `HIP_VISIBLE_DEVICES=$IDX` for AMD, `ONEAPI_DEVICE_SELECTOR=level_zero:<IDX>` for Intel — colon syntax, required by the current oneAPI/SYCL runtime (older slash-based forms abort with "Incomplete selector!" / "Backend is required but missing") and restrict docker GPU passthrough to that device (`--gpus 'device=$IDX'` for NVIDIA; AMD/Intel pass-through unchanged, env var selects).

> **Image entrypoint gotcha (verified v0.28.0, all three official images):**
> the images set `ENTRYPOINT ["vllm", "serve"]`. `docker run IMAGE bash -s` would
> therefore execute `vllm serve bash -s` — and `-s` abbreviates to `-sc`
> (`--speculative-config`), failing with `argument --speculative-config/-sc:
> expected one argument`. Always pass `--entrypoint bash` (done in spike.sh).

### Metadata captured (→ `environment.json`)

- GPU model, VRAM total, driver version, (CUDA / ROCm / XPU stack versions)
- Kernel, OS release, kernel GPU modules (`nvidia` / `amdgpu` / `xe` / `i915`)
- CPU model + core count, host RAM
- Docker version, **image name + pinned tag + image digest**
- vLLM version (`vllm --version` in container)
- **Visible GPU in container** (runtime-verified: `nvidia-smi` / `rocm-smi --showproductname` / `torch.xpu.device_count()` + `torch.xpu.get_device_name()` on XPU; confirms the host-side largest-VRAM selection actually landed)

---

## 4. Inference Stack: vLLM — pinned

**Version pinned: `v0.28.0`** (current vLLM stable, 2026-08-26) — **same tag
exists in all three official repos**, verified on Docker Hub:

| Vendor | Image                          | Tag       | Size   |
|--------|--------------------------------|-----------|--------|
| NVIDIA | `vllm/vllm-openai`             | `v0.28.0` | ~9.7 GB |
| AMD    | `vllm/vllm-openai-rocm`        | `v0.28.0` | ~11.4 GB |
| Intel  | `vllm/vllm-openai-xpu`         | `v0.28.0` | ~4.1 GB |

`--image vllm/vllm-openai:v0.28.0` style override allowed, but pinned is the default.
(Images are amd64; ARM out of scope.)

Server command shape (inside container):

```
vllm serve <HF_MODEL> \
  --host 0.0.0.0 --port 8000 \
  --max-model-len <per-model> \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  <per-config flags>
```

---

## 5. Model Matrix (target VRAM: 32–40 GB) — latest generations (Sept 2026)

All 4 models must fit the **smallest target card (32 GB)** — weights + KV cache
+ overhead — so results are comparable across the whole fleet. (40 GB cards
simply get more KV headroom.) This forces quantized checkpoints for the
25–35 B class. Verified on HuggingFace today (on-disk sizes measured from the
repos' safetensors):

| # | Slot            | Model (HF repo)                                   | Format        | On disk | KV headroom @32 GB (0.90 util) | Gated |
|---|-----------------|---------------------------------------------------|---------------|---------|--------------------------------|-------|
| M1 | dense ~8–9 B   | `Qwen/Qwen3.5-9B` (9.65 B)                        | BF16          | 19.3 GB | ~9.5 GB  | no (12.6 M dl) |
| M2 | MoE small      | `openai/gpt-oss-20b` (21 B total / 3.6 B active)   | MXFP4 native  | 13.8 GB | ~15 GB   | no |
| M3 | dense 25–30 B  | `Qwen3.8-27B` AWQ (27.8 B, newest 27 B gen)        | AWQ 4-bit     | 21.0 GB  | ~7.8 GB | no (1.08 M dl) |
| M4 | MoE 25–35 B    | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (35.95 B / 3 B active, official) | GPTQ 4-bit | 24.4 GB | ~4.4 GB | no (491 k dl) |

Selection rationale (latest generation, Sept 2026):
- **M1:** `Qwen3.5-9B` is the newest Qwen dense in the 8–9 B class (Qwen3.5 gen).
  BF16; on 32 GB leaves ~9.5 GB KV headroom, on 40 GB ~16.7 GB.
- **M2:** no *newer* small MoE exists in 2026 (Nemotron small MoE = 4 B, too
  small; Qwen3.8-Flash-Next = 180 B, too big). `gpt-oss-20b` remains the small-MoE
  reference and keeps vLLM first-class native-format support.
- **M3:** the newest 27 B dense is `Qwen/Qwen3.8-27B` (5 M dl). Its official
  FP8 (27.8 GB) leaves **no** KV room on 32 GB → use the 4-bit AWQ:
  default `cyankiwi/Qwen3.8-27B-AWQ-INT4` (21.0 GB, 1.08 M dl). (AMD's
  `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`, 19.5 GB, available via `--model`.)
  **Hybrid GDN architecture:** every 4th layer uses full attention (KV cache);
  the remaining 3/4 use linear attention (fixed-size recurrent state per
  layer). This reduces KV footprint to ~1/4 of a dense model.
  On 32 GB at 0.90 util this OOMs at startup (vLLM v0.28 counts CUDA-graph
  memory against the util budget): `Available KV cache memory: -X GiB`.
  Fit flags: 0.95 util + 2048 max_model_len + 2048 max-num-batched-tokens.
  `long-context` (32k) is dropped: a 21 GB model cannot hold 32k context on
  32 GB, even at fp8 KV.
  M3 baseline at 0.95 util: pool ~3.3 GiB ≈ 36k tokens — enough for C=16
  workload (12.3k tokens) + GDN state (1.2 GB @ C=16).
- **M4:** `Qwen/Qwen3.5-35B-A3B` (35.95 B / 3 B active). **Cross-vendor
  checkpoint: AWQ-4-bit** `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` (24.5 GB) —
  the original official GPTQ-Int4 (`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`) is NOT
  supported by the vLLM ROCm backend (verified on the 2026-09-02 ROCm 7.2.3
  run: both M4 cells `skipped:unsupported:<…>`; the server rejects the GPTQ
  attention/expert weights at load time). The AWQ variant is the same
  compressed-tensors pack-quantized int4 family M3 uses on ROCm, so it keeps
  cross-vendor comparability. The GPTQ id is retained (commented) for
  NVIDIA-only runs. Heaviest cell (24.5 GB): needs the 0.95 util / 2048-ctx /
  2048-batch fit flags; `--validate` first.
- **Execution model: each run targets one machine** (a different GPU per test).
  The report is therefore a **standalone per-machine deliverable**: it must be
  self-contained (full environment block, image digest, vLLM version) and no
  cross-run/diff mode is needed. The host HF cache (`~/.gpu-bench/hf-cache`,
  bind-mounted) only matters for re-runs/debug on the same machine.
- `--model-id` can add/replace entries; models download to that cache
  (`HF_HOME=/hf-cache`).

---

## 6. Per-Model Optimization Configs

Each (model, config) pair = **one server process** with distinct flags, then the
concurrency sweep. Matrix defined in `config/models.yaml`:

| Model | Configs tested |
|-------|----------------|
| M1 Qwen3.5-9B (dense)       | `baseline` · `kv-fp8` (`--kv-cache-dtype fp8`) · `long-context` (`--max-model-len 32768`, chunked prefill default on) |
| M2 gpt-oss-20b (MoE)        | `baseline` · `kv-fp8` |
| M3 Qwen3.8-27B-AWQ (dense)  | `baseline` · `kv-fp8` · `long-context` |
| M4 Qwen3.5-35B-A3B-GPTQ (MoE) | `baseline` · `kv-fp8` |

Speculative decoding is **not tested.** MTP is model-specific (`qwen3_5_mtp`,
`deepseek_mtp`, …) and unsupported for every matrix model in v0.28.0 — gpt-oss
`GptOssForCausalLM` has no MTP handler (`--speculative-config.method mtp` →
`NotImplementedError` at arg validation, verified in spike) and the Qwen3.5-35B
base has no MTP head (`num_nextn_predict_layers` absent from config.json). The
only generic method, `ngram`, was verified working in the spike but is not a
meaningful perf axis on the random workload, so the MoE models run
`baseline · kv-fp8` only.

- **Auto-skip semantics:** if the vendor backend rejects a config (e.g. FP8 KV
  cache not supported on XPU/ROCm in v0.28.0, MTP kernel unavailable), the server
  fails to start; the orchestrator detects it (health-timeout + log pattern),
  marks the cell `skipped:<reason>` in the report, and continues. The report
  shows the full matrix with pass/skip.
- Each config row also records the **effective server flags** used.
- Common server settings: `--gpu-memory-utilization 0.90`,
  `--enable-prefix-caching` off (it distorts synthetic-token throughput;
  re-enable only in a dedicated `prefix-cache` config if wanted).

### Model lifecycle: download → test → **keep** weights (re-run speed)

Weights are **kept** in the HF cache after each model finishes, so re-runs
skip the ~20–25 GB re-download.  This speeds up iterative runs (A-B testing,
config changes, re-validating M3/M4 after a fix).

```
for model in M1..M4:
    hf_download(model)                          # into /hf-cache (bind mount); no-op if cached
    for config in model.configs:
        run server + concurrency sweep cells
    # weights KEPT (no deletion) — re-run skips download
```

- **`./clean.sh`** — standalone host script to remove cached weights
  (`./clean.sh` for all, `./clean.sh M3,M4` for specific, or by repo ID).
- **`--delete-weights`** (on `bench.sh`) restores the old per-model deletion
  when disk is very tight.  `--keep-weights` remains a **no-op** for
  backward compatibility.
- Peak disk = image ~12 GB + **all** weights ~78.6 GB + logs ≈ 100 GB on the
  first run (≈ 40 GB with `--delete-weights`).  Preflight gates ~120 GB
  (shared-FS) / ~85 GB (model-only) for keep, 65 GB for delete-weights.
- `report.json` records per-model weights size and a `weights_removed:
  true/false` flag (false by default now).

### Total run estimate (full matrix = default)

13 (model,config) cells × (model load ~2–8 min + 4-level sweep ~10–15 min)
≈ **3.5–4.5 h** per machine — this is the **default**. `--quick` (M1 only,
baseline + kv-fp8, 2 concurrency levels) remains for pre-flight / smoke tests.

---

## 7. Benchmark Workload

Driven entirely by `vllm bench serve` (one invocation per cell × concurrency):

```
vllm bench serve \
  --host 127.0.0.1 --port 8000 \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model <HF_MODEL> \
  --dataset-name random \
  --random-input-len 512 --random-output-len 256 \
  --num-prompts 50 \
  --max-concurrency <1|4|8|16> \
  --num-warmups 2 \
  --seed 42 \
  --temperature 0 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,90,99 \
  --save-result --result-filename <cell>_<C>.json
```

> **v0.28.0 `bench serve` flag facts (verified in spike, all three official images):**
> - Backend is **`openai-chat`** — the old `vllm-chat-completions` name is gone.
>   The `openai-chat` backend requires the URL to end in `chat/completions`, so
>   pass **`--endpoint /v1/chat/completions`** explicitly (the v0.28 default
>   `--endpoint` is `/v1/completions`, which the chat backend rejects).
> - Plain `--help` is grouped by config; use **`--help=all`** to list every flag.
> - **`--num-warmups N`** exists (default 0) — use it for load stabilization.

- Concurrency levels: **1, 4, 8, 16** (16 is the cap; more would pressure KV on 32 GB)
- 50 prompts/level, **`--num-warmups 2`** for load stabilization (flag verified
  available in v0.28.0; warmup requests are excluded from the reported metrics)
- `--ignore-eos` + `temperature 0` → fixed 256-token outputs for stable tok/s
- Workload shape identical across all 4 models and all vendors (comparability)
- `--input-len/--output-len` and `--num-prompts` overridable via CLI/env

**Metrics per cell × concurrency** (from `vllm bench serve` JSON — actual v0.28.0 keys, captured in spike):

| Metric | JSON key(s) |
|--------|-------------|
| TTFT ms | `p50_ttft_ms`, `p90_ttft_ms`, `p99_ttft_ms` (also `mean/median/std_ttft_ms`) |
| TPOT ms | `p50_tpot_ms`, `p90_tpot_ms`, `p99_tpot_ms` (also `mean/median/std_tpot_ms`) |
| ITL ms  | `p50_itl_ms`, `p90_itl_ms`, `p99_itl_ms` (also `mean/median/std_itl_ms`) |
| Request throughput (req/s) | `request_throughput` |
| Output token throughput (tok/s) | `output_throughput` |
| Total token throughput (tok/s) | `total_token_throughput` |
| Peak output tok/s | `max_output_tokens_per_s` |
| Real-time factor | `rtfx` |
| Wall time (s) | `duration` |
| Completed / failed / total | `completed`, `failed`, `num_prompts` |
| Request rate | `request_rate` |
| Goodput | `request_goodput` |
| Token totals | `total_input_tokens`, `total_output_tokens` |

`report.py` maps these flat keys into the normalized `metrics{}` object in the
§8 schema (e.g. `p50_ttft_ms` → `metrics.ttft_ms.p50`).

**Plus our telemetry** (1 Hz sampler in the container, per-vendor):
GPU memory used (peak), GPU utilization (% avg), and **best-effort power draw**
(W avg) via `nvidia-smi` / `rocm-smi` / `intel_gpu_top --json` — where the
vendor tool lacks a power field, the report carries `null` ("n/a") rather than
failing.

---

## 8. Report Output

`results/<YYYYMMDD-HHMMSS>_<gpu-slug>/` — e.g.
`results/20260902-160512_nvidia-geforce-rtx-4090/`

- **One sub-directory per run**, created *in-container* by `run_matrix.py`
  right after GPU selection, so the GPU model name in the slug is the card
  actually benchmarked. The timestamp is the host start time (bench.sh passes
  `RUN_ID`); a collision (same second + same GPU) is disambiguated with a
  `-2`, `-3`, … suffix. Repeated runs therefore never overwrite each other.
- The run-id basename is written to `results/.latest`, which `entrypoint.sh`
  (in-container report step) and `bench.sh` (final report echo) use to locate
  the current run. `report.json`/`report.md` metadata carries `run_id` + `gpu`.

| File                | Content                                              |
|---------------------|------------------------------------------------------|
| `report.md`         | Per-model tables (rows = configs, cols = concurrency levels, cells = TTFT/TPOT/tok-s/util), skip reasons, environment block, image digest |
| `report.json`       | Full machine-readable dataset (see schema)            |
| `environment.json`  | Host + GPU + image + vLLM versions                    |
| `server_<model>_<config>.log` | vLLM server logs (per cell)              |
| `bench_<model>_<config>_<C>.json` | raw `vllm bench serve` outputs       |
| `telemetry_<model>_<config>.json` | 1 Hz GPU samples (mem/util/power)        |

### `report.json` schema

```json
{
  "schema_version": "1.0",
  "run_id": "20260902-120000_nvidia-geforce-rtx-4090",
  "generated_at": "2026-09-02T16:05:12+00:00",
  "environment": {
    "vendor": "nvidia",
    "gpu": "NVIDIA GeForce RTX 4090",
    "gpu_index_in_container": 0,
    "vram_total_gb": 24.0,
    "driver": "570.124.06",
    "stack": {"cuda": "12.9", "rocm": null, "xpu": null},
    "os": "Ubuntu 24.04", "kernel": "6.8.0-xx",
    "cpu": "AMD EPYC ...", "cpu_cores": 128, "ram_gb": 512.0,
    "gpu_kernel_modules": ["nvidia", "nvidia_drm", "nvidia_modeset", "nvidia_uvm"],
    "docker_version": "27.3.1", "image": "vllm/vllm-openai:v0.28.0",
    "image_id": "sha256:...", "vllm_version": "0.28.0"
  },
  "workload": {"random_input_len": 512, "random_output_len": 256, "num_prompts": 50,
               "num_warmups": 2, "concurrency_levels": [1,4,8,16], "seed": 42, "...": "..."},
  "models": {
    "M1": {"id": "Qwen/Qwen3.5-9B", "class": "dense-9b", "format": "bf16",
           "weights_gb": 19.3, "max_model_len": 8192,
           "configs": ["baseline", "kv-fp8", "long-context"]},
    "M2": {"id": "openai/gpt-oss-20b", "...": "..."},
    "M3": {"id": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "...": "..."},
    "M4": {"id": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", "...": "..."}
  },
  "metadata": {"gpu": "NVIDIA GeForce RTX 4090", "total_cells": 13, "total_rows": 52,
               "bench_rows": 50, "skip_rows": 2, "fail_rows": 0, "vllm_version": "0.28.0"},
  "rows": [
    {
      "model": "Qwen/Qwen3.5-9B", "config": "baseline", "concurrency": 8,
      "status": "ok", "reason": null,
      "completed": 50, "failed": 0,
      "request_throughput": 0.88, "output_throughput": 225.1,
      "ttft_p50_ms": 41.2, "ttft_p90_ms": 95.0, "ttft_p99_ms": 180.4,
      "tpot_p50_ms": 9.1,  "tpot_p90_ms": 12.3, "tpot_p99_ms": 21.0,
      "itl_p50_ms": 9.0,   "itl_p90_ms": 12.0,  "itl_p99_ms": 19.5,
      "duration_s": 56.9,
      "telemetry": {"mem_peak_gb": 26.1, "util_avg_pct": 96.0, "power_avg_w": 335.0, "...": "..."}
    }
  ]
}
```

`rows` is one entry per (cell × concurrency) — flat, with the raw v0.28 bench
keys mapped through `report.py: BENCH_KEY_MAP`; cell-level skips/failures
repeat the status/reason across the cell's concurrency levels with null
metrics. `environment` is `null` if collection failed (never fatal).

`report.md` renders: header (GPU, run-id, vLLM version), metric counts,
**Environment** table, C=1 model summary, then one table per model — rows =
config × concurrency, columns = status + latency/throughput + GPU telemetry:

```markdown
# GPU Inference Bench Report

**NVIDIA GeForce RTX 4090** · Run: 20260902-160512_nvidia-geforce-rtx-4090 · vLLM 0.28.0 · ...

## Environment
| Field | Value |
|-------|-------|
| GPU   | NVIDIA GeForce RTX 4090 |
| Image | vllm/vllm-openai:v0.28.0 |
| Image ID | sha256:... |

## M4 · Qwen/Qwen3.5-35B-A3B-GPTQ-Int4

| Config | C | Status | Req/s | Tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | Dur s | Mem peak GB | Util % | Power W |
|--------|---|--------|-------|-------|----------|----------|----------|----------|---------|---------|-------|-------------|--------|---------|
| baseline | 1 | ok | 0.15 | 37 | 38 ms | 71 ms | ... |
| kv-fp8   | 1 | skipped: kv-fp8-unsupported | n/a | n/a | ... |
```

---

## 9. `bench.sh` Flow

```bash
#!/usr/bin/env bash
set -euo pipefail

# Flags:
#   --models M1,M2,M3,M4 | --configs baseline,kv-fp8,long-context | --concurrency 1,4,8,16
#   --image <override>   --vendor <nvidia|amd|intel>              --quick
#   --delete-weights (default: keep weights for re-runs; clean.sh to free disk)
#   --cache-dir <dir>    --results <dir>  --gpu-index N  --start-timeout S
#   --version <tag>      --dry-run        --force

# 1. preflight: docker daemon, vendor detected, disk gates (shared-FS aware),
#    stale container/image cleanup
# 2. host metadata (host OS, docker version, image id) → env for container
# 3. IMAGE=<(vendor → vllm/vllm-openai[-rocm|-xpu]:$VLLM_VERSION, or --image);
#    docker pull + post-pull disk gate
# 4. docker run --rm --entrypoint bash (override image ENTRYPOINT ["vllm","serve"]!)
#    <vendor device flags> -e RUN_ID=<host timestamp>
#      -v $REPO_DIR   → /bench (ro: container/ + config/models.yaml)
#      -v $HF_CACHE   → /hf-cache (rw)
#      -v $RESULTS   → /results (rw)
#      $IMAGE /bench/container/entrypoint.sh
# 5. tail report.md of the run dir (via results/.latest);
#    non-zero exit on any failed cell (skip ≠ failure)
```

Container `entrypoint.sh` → `run_matrix.py` implements the §6 loop with:
- in-container GPU selection (max-VRAM or `--gpu-index`) + GPU name capture,
- `environment.json` collection (best-effort, never fatal),
- per-run output dir `<results>/<RUN_ID>_<gpu-slug>/` + `.latest` pointer,
- health wait (default 900 s, `--start-timeout`), startup-log parse for skip reasons,
- GPU telemetry thread (vendor-specific 1 Hz sampler),
- `subprocess` calls to `vllm serve` / `vllm bench serve`,
- weights KEPT after each model (default; `--delete-weights` to delete),
- `report.py` aggregation at the end (even on partial failure).

---

## 10. Error Handling

| Scenario                                  | Behavior                                                        |
|-------------------------------------------|------------------------------------------------------------------|
| No GPU / vendor                            | Exit with install hint (`nvidia-container-toolkit`, `amdgpu` fw, Intel XPU driver) |
| Image pull fails                           | Exit (network)                                                   |
| Model download fails / gated               | Exit with `hf token` guidance; M1–M4 defaults are all ungated    |
| Server OOM at startup                      | Cell marked `skipped:oom` (or `failed` if unexpected); continue  |
| Config unsupported on backend (FP8 KV / MTP) | Cell marked `skipped:<log-derived reason>`; continue          |
| Benchmark run errors (>5 % failed reqs)    | Cell marked `failed`, raw JSON kept; report shows failure        |
| Any cell failure                           | Exit code 1 at the end, but report is always written             |

---

## 11. Open Decisions

**Resolved:** M3 = cyankiwi AWQ · **M4 = cyankiwi Qwen3.5-35B-A3B AWQ-4bit**
(GPTQ-Int4 not supported on ROCm — verified 2026-09-02; GPTQ id kept as
NVIDIA-only comment) ·
VRAM floor 32 GB (32–40 GB fleet) · **full matrix is the default (~4 h)** ·
**weights kept after each model** (`--delete-weights` to delete; `clean.sh`
to free disk) ·
power metrics **best-effort** (null when the vendor tool lacks them) ·
per-machine standalone reports (no cross-run diff mode) ·
**MTP dropped** — not supported for any matrix model in v0.28.0 (spike); the
two MoE models run **`baseline · kv-fp8` only** (no speculative decoding — `ngram`
was benchmarked in the spike but is not a meaningful perf axis on the random
workload) · **M3 `long-context` dropped** (32 k ctx can't fit a 21 GB model on
32 GB; M1 keeps it) · **`--validate` preflight** (static estimate + live
probe; `container/validate_fit.py`) catches startup OOM / unsupported-quant
before a full ~4 h run.

**No other open items — ready to implement.**

---

## 12. Milestones

| # | Task | Notes |
|---|------|-------|
| 1 | Plan v3.1 review (this doc) | done |
| 2 | **Spike:** verify `vllm bench serve` JSON schema, speculative-config flags, `--kv-cache-dtype fp8` support, `--num-warmups` availability in v0.28.0 | **done** — `spike.sh`; JSON schema captured, fp8 KV OK, ngram OK, MTP unsupported for gpt-oss, backend=`openai-chat`+`--endpoint` |
| 3 | `config/models.yaml` + `run_matrix.py` (server lifecycle, sweep, auto-skip) | **done** — in-container GPU selection (AMD/`rocm-smi` max-VRAM or `--gpu-index`), weight download/delete, health-wait, skip-reason parse, C-sweep, `cells.json` |
| 4 | `telemetry.py` (nvidia-smi / rocm-smi / intel_gpu_top samplers) | **done** — 1 Hz sampler; AMD reads unfiltered `rocm-smi` by physical idx; NVIDIA `nvidia-smi -i`; Intel `xpu-smi` best-effort; `start/stop/aggregate` API |
| 5 | `report.py` (JSON + Markdown) | **done** — maps real v0.28 bench keys (`p50_ttft_ms`…) → normalized schema; per-(cell,C) rows; `report.json` + `report.md` |
| 6 | `bench.sh` (detection, pull, run, exit codes) | **done** — vendor detect, per-vendor image + GPU args, disk gates, stale cleanup, `docker run --entrypoint bash` → `entrypoint.sh`, `--dry-run` |
| 7 | Smoke test: `--quick` on NVIDIA (this env or user's box) | pending — needs a live GPU box (this repo was built on a GPU-less host) |
| 8 | Full matrix run, NVIDIA | pending — needs a live GPU box |
| 9 | Port checks: AMD, Intel (XPU FP8-KV/MTP auto-skip paths exercised) | pending — needs live AMD/Intel boxes |
| 10 | README + usage examples | **done** |
| 11 | Live ROCm validation & M3/M4 fix | **done** — M3 fit flags, M4 AWQ swap, `--validate` preflight |
| 12 | Weight retention for re-runs | **done** — weights kept by default, `clean.sh` standalone, `--delete-weights` opt-in, dynamic disk gate |
| 12 | Weight retention for re-runs | **done** — weights kept by default, `clean.sh` standalone, `--delete-weights` opt-in, dynamic disk gate |

---

*Draft v3.1 — 2026-09-02 (final: cyankiwi 27B, Qwen 35B cross-vendor, 32–40 GB fleet, full-matrix default, per-model weight deletion, best-effort power metrics, per-machine standalone reports; latest-gen models; measured on-disk sizes; MTP pre-check; NVFP4 cross-vendor exclusion)*
*
*v3.2 — 2026-09-02 (spike run 1 on ROCm 7.2.3 target: MTP not supported for gpt-oss-20b in vLLM 0.28.0 → M2 spec-mtp now auto-skipped; open item: spec-mtp column vs spec-ngram; bench JSON schema + fp8-KV support pending spike run 2)*
*
*v3.3 — 2026-09-02 (spike complete on ROCm 7.2.3 target: bench backend is `openai-chat`+`--endpoint /v1/chat/completions`, `--help=all`, `--num-warmups` available, fp8 KV + ngram verified working. MTP dropped from all matrix models. §7 bench command corrected.)*
*
*v3.4 — 2026-09-02 (MoE models M2/M4 run `baseline · kv-fp8` only — no speculative decoding; real bench result JSON schema captured from spike, recorded as the report.py field source.)*
*
*v3.5 — 2026-09-02 (milestones 3–6 built & mock-tested end-to-end: `config/models.yaml`, `container/run_matrix.py` (in-container GPU select, server lifecycle, C-sweep, weight delete, `cells.json`), `container/telemetry.py` (1 Hz AMD/NVIDIA/Intel), `container/report.py` (real-key mapping, JSON+MD), `container/entrypoint.sh`, `bench.sh` (vendor detect, image+GPU args, disk gates, stale cleanup, `--dry-run`). Full 4-model mock matrix: 10 cells → 20 rows, clean. Next: milestone 7 (real `--quick` smoke on a live GPU).)*
*
*v3.6 — 2026-09-02 (implementation finished for the GPU-less build host: per-run output dirs `results/<ts>_<gpu-slug>/` + `.latest`; `environment.json` (GPU/driver/stack/OS/kernel/CPU/RAM/docker/image-id/vLLM) written in-container, best-effort; `cells.json` is now a self-contained manifest (workload/models/cells); `report.json` schema v1.0 (run_id, environment, workload, models, metadata, flat rows); `report.md` env block + per-model tables; `bench.sh` gains `--image`/`--cache-dir`, passes host OS/docker-version/image-id to the container. Milestone 10 done (README). Milestones 7–9 remain: they require live GPU hardware.)*
*
*v3.7 — 2026-09-03 (live ROCm 7.2.3 validation, 32 GB card 0x7551, run 20260902-220721. **M3** `skipped:oom` root cause: vLLM v0.28.0 counts CUDA-graph memory against `--gpu-memory-utilization` (default since v0.21.0), so 21 GB weights + ~4.5–7 GB graph/profiling at 8192 ctx exceeds the 0.90×31.9 GB=28.7 GB budget → `Available KV cache memory: −X GiB` → `No available memory for the cache blocks`. Fix: model-level `gpu_memory_utilization 0.95` + `max_model_len 2048` (workload needs 768) + `max-num-batched-tokens 2048`; `long-context` dropped (32 k ctx cannot fit on 32 GB). **M4** `skipped:unsupported:<…>` root cause: GPTQ-Int4 is not supported by the vLLM ROCm backend (server rejects GPTQ attention/expert weights at load). Fix: M4 → `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` (same compressed-tensors int4 family as M3; GPTQ id kept as NVIDIA-only comment). New **`--validate` preflight** (`container/validate_fit.py`): static HF-based estimate (hybrid-GDN KV aware) + live server probe that parses vLLM's own sizing lines for a definitive FIT/TIGHT/NO-FIT/UNSUPPORTED verdict; overhead auto-calibrated from the previous run's telemetry. `run_matrix.build_server_cmd` gains model-level `flags` + `gpu_memory_utilization` override (common < model < config). `gpu_name()` strips rocm-smi's raw `Card Model:` prefix. Milestone 11 done.)*
*
*v3.8 — 2026-09-03 (weight retention for faster re-runs. Weights are now KEPT in the HF cache after each model (re-runs skip the ~20–25 GB re-download); `--keep-weights` is a deprecated no-op. New **`clean.sh`** host script removes cached weights: `./clean.sh` (all), `./clean.sh M3,M4` (by key), or by HuggingFace repo ID; `--dry-run` previews. **`bench.sh --clean [M1,M2,...]`** is a thin wrapper (cleans + exits, no container). **`bench.sh --delete-weights`** restores the old per-model deletion (peak disk back to ~40 GB). Disk gate is now dynamic: NEED_MODEL_GB=85 (keep, whole weight set) by default, 30 (delete-weights). `run_matrix` takes `--delete-weights` and sets `weights_removed:false` in cells by default. Milestone 12 done.)*