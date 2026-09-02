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
├── results/                  # output (gitignored)
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
| Intel   | `lspci \| grep -iE 'intel.*(vga|3d)'` or `/dev/dri/renderD*` | `--device=/dev/dri --group-add=video`                                             |

No vendor detected → exit with hint. `--vendor` flag can force.
Single-GPU scope: index 0 only (`CUDA_VISIBLE_DEVICES=0` / `HIP_VISIBLE_DEVICES=0` / `ONEAPI_DEVICE_SELECTOR=0`).

### Metadata captured (→ `environment.json`)

- GPU model, VRAM total, driver version, (CUDA / ROCm / XPU stack versions)
- Kernel, OS release, kernel GPU modules (`nvidia` / `amdgpu` / `xe` / `i915`)
- CPU model + core count, host RAM
- Docker version, **image name + pinned tag + image digest**
- vLLM version (`vllm --version` in container)

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
- **M4:** `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (official, cross-vendor GPTQ 4-bit
  supported on CUDA + ROCm + XPU). The newer NVIDIA Nemotron-3.5 30B-A3B is
  NVFP4 — an NVIDIA-specific format that would auto-skip on ROCm/XPU and break
  cross-vendor comparability → excluded from the default, available via `--model`
  for NVIDIA-only runs.
  Heaviest cell (24.4 GB): C=16 KV (~2.3 GB at 512+256, bf16 KV) fits on 32 GB
  (~4.4 GB headroom); `kv-fp8` halves the KV need.
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
| M2 gpt-oss-20b (MoE)        | `baseline` · `kv-fp8` · `spec-mtp` (native MTP speculative decoding²) |
| M3 Qwen3.8-27B-AWQ (dense)  | `baseline` · `kv-fp8` · `long-context` |
| M4 Qwen3.5-35B-A3B-GPTQ (MoE) | `baseline` · `kv-fp8` · `spec-mtp` (auto-skipped — no MTP head in config³) |

² Only `gpt-oss-20b` in the matrix has a verified MTP head
   (`GptOssForCausalLM`, natively supported by the vLLM gpt-oss backend).
   Exact flag to verify in spike: `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1, ...}'`
   and record the effective flags in the report.

³ Pre-check: the orchestrator reads the HF `config.json` of each model before
   starting a server; `num_nextn_predict_layers` absent ⇒ `spec-mtp` marked
   `skipped:no-mtp-head` **without launching the server** (no wasted minutes).
   Qwen3.5-35B-A3B base has no nextn layers; if an Instruct variant with MTP
   appears, the config file picks it up automatically.

- **Auto-skip semantics:** if the vendor backend rejects a config (e.g. FP8 KV
  cache not supported on XPU/ROCm in v0.28.0, MTP kernel unavailable), the server
  fails to start; the orchestrator detects it (health-timeout + log pattern),
  marks the cell `skipped:<reason>` in the report, and continues. The report
  shows the full matrix with pass/skip.
- Each config row also records the **effective server flags** used.
- Common server settings: `--gpu-memory-utilization 0.90`,
  `--enable-prefix-caching` off (it distorts synthetic-token throughput;
  re-enable only in a dedicated `prefix-cache` config if wanted).

### Model lifecycle: download → test → delete (storage bloat prevention)

Test machines are expected to have limited disk, so weights are **not** kept
across models:

```
for model in M1..M4:
    hf_download(model)                          # into /hf-cache (bind mount)
    for config in model.configs:
        run server + concurrency sweep cells
    rm -rf /hf-cache/hub/models--<org>--<name>  # after ALL its cells done
```

- Deletion runs after the model's last cell (ok **or** failed), with a
  `--keep-weights` escape hatch for debugging. Weights are also kept
  automatically when the run was started with `--models <single>` (so a
  re-run of just that model doesn't re-download).
- Failure to delete → **warn and continue** (never fatal).
- Peak disk = image ~12 GB + largest weights 24.4 GB + logs ≈ 40 GB →
  preflight requires **50 GB free** (vs ~120 GB if all weights were retained).
- `report.json` records per-model weights size and a `weights_removed: true/false` flag.

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
  --backend vllm-chat-completions \
  --model <HF_MODEL> \
  --dataset-name random \
  --random-input-len 512 --random-output-len 256 \
  --num-prompts 50 \
  --max-concurrency <1|4|8|16> \
  --seed 42 \
  --temperature 0 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,90,99 \
  --save-result --result-filename <cell>_<C>.json
```

- Concurrency levels: **1, 4, 8, 16** (16 is the cap; more would pressure KV on 32 GB)
- 50 prompts/level, warm-up handled by `vllm bench serve` (first requests after server start
  are part of load stabilization; optional `--num-warmups` if available in v0.28.0 — verify in spike)
- `--ignore-eos` + `temperature 0` → fixed 256-token outputs for stable tok/s
- Workload shape identical across all 4 models and all vendors (comparability)
- `--input-len/--output-len` and `--num-prompts` overridable via CLI/env

**Metrics per cell × concurrency** (from `vllm bench serve` JSON):
TTFT p50/p90/p99, TPOT p50/p90/p99, ITL p50/p90/p99, request throughput (req/s),
output token throughput (tok/s), total wall time, successful/failed count.

**Plus our telemetry** (1 Hz sampler in the container, per-vendor):
GPU memory used (peak), GPU utilization (% avg), and **best-effort power draw**
(W avg) via `nvidia-smi` / `rocm-smi` / `intel_gpu_top --json` — where the
vendor tool lacks a power field, the report carries `null` ("n/a") rather than
failing.

---

## 8. Report Output

`results/<YYYYMMDD-HHMMSS>_<vendor>_<gpu-slug>/`

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
  "schema_version": "2.0",
  "run_id": "20260902-120000_nvidia_rtx4090",
  "environment": {
    "vendor": "nvidia",
    "gpu": "NVIDIA GeForce RTX 4090",
    "vram_total_gb": 24,
    "driver": "570.124.06",
    "stack": {"cuda": "12.9", "rocm": null, "xpu": null},
    "os": "Ubuntu 24.04", "kernel": "6.8.0-xx",
    "image": "vllm/vllm-openai:v0.28.0",
    "image_digest": "sha256:...",
    "vllm_version": "0.28.0"
  },
  "workload": {"input_tokens": 512, "output_tokens": 256,
               "concurrency_levels": [1,4,8,16], "num_prompts": 50, "seed": 42},
  "models": {
    "M1": {"id": "Qwen/Qwen3.5-9B",  "class": "dense-9b",  "format": "bf16",   "weights_gb": 19.3},
    "M2": {"id": "openai/gpt-oss-20b", "class": "moe-small", "format": "mxfp4",  "weights_gb": 13.8},
    "M3": {"id": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "class": "dense-27b", "format": "awq4", "weights_gb": 21.0},
    "M4": {"id": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", "class": "moe-35b-a3b", "format": "gptq4", "weights_gb": 24.4}
  },
  "results": [
    {
      "model": "M1", "config": "baseline",
      "server_flags": {"max_model_len": 8192},
      "status": "ok",
      "concurrency": 8,
      "metrics": {
        "ttft_ms": {"p50": 41.2, "p90": 95.0, "p99": 180.4},
        "tpot_ms": {"p50": 9.1,  "p90": 12.3, "p99": 21.0},
        "itl_ms":  {"p50": 9.0,  "p90": 12.0, "p99": 19.5},
        "request_throughput_rps": 0.88,
        "output_token_throughput_tps": 225.1,
        "wall_time_s": 56.9
      },
      "gpu": {"mem_peak_gb": 26.1, "util_avg_pct": 96.0, "power_avg_w": 335.0,
               "weights_removed": true}
    }
  ]
}
```

`report.md` renders one table per model: rows = configs (with skip reasons),
columns grouped by concurrency:

```markdown
### M4 · Qwen3-30B-A3B (AWQ-4bit, 17 GB)

| config    | C  | TTFT p50 | TTFT p99 | TPOT p50 | req/s | tok/s | GPU mem | util |
|-----------|----|----------|----------|----------|-------|-------|---------|------|
| baseline  |  1 |  38 ms   |  71 ms   |  6.9 ms  | 0.15  | 37    | 19.2 GB |  88% |
|           | 16 | ...      | ...      | ...      | ...   | ...   | ...     | ...  |
| kv-fp8    | 16 | ...      | ...      | ...      | ...   | ...   | ...     | ...  |
| mtp       | 16 | ...      | ...      | ...      | ...   | ...   | ...     | ...  |
| mtp       | —  | SKIPPED: qwen3_mtp not supported on this backend |
```

---

## 9. `bench.sh` Flow

```bash
#!/usr/bin/env bash
set -euo pipefail

# Flags:
#   --models M1,M2,M3,M4 | --configs baseline,kv-fp8,mtp | --concurrency 1,4,8,16
#   --image <override>   --vendor <nvidia|amd|intel>      --quick
#   --keep-weights (debug; default: delete after each model)
#   --cache-dir ~/.gpu-bench/hf-cache  --results-dir ./results
#   --keep-server (debug) --dry-run

# 1. preflight: docker daemon, vendor detected, disk ≥ 50 GB free (sequential
#    weights, deleted per model), nvidia-container-toolkit (nvidia only)
# 2. detect vendor (lspci / /dev/kfd / /dev/dri); collect environment.json
# 3. IMAGE=<(vendor → vllm/vllm-openai[-rocm|-xpu]:v0.28.0); docker pull
# 4. docker run --rm <device flags> --shm-size 16g
#      -v benchmark container/ → /bench (ro)
#      -v config/models.yaml   → /bench/config (ro)
#      -v $CACHE_DIR           → /hf-cache (rw)
#      -v $RESULTS_DIR         → /results (rw)
#      $IMAGE bash /bench/entrypoint.sh   # env: MODEL list, CONFIGS, CONCURRENCY...
# 5. tail report.md path; non-zero exit on any failed cell (skip ≠ failure)
```

Container `entrypoint.sh` → `run_matrix.py` implements the §6 loop with:
- health wait (≤ 300 s), startup-log parse for skip reasons,
- GPU telemetry thread (vendor-specific sampler),
- `subprocess` calls to `vllm serve` / `vllm bench serve`,
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

**Resolved:** M3 = cyankiwi AWQ · M4 = Qwen 35B-A3B GPTQ-Int4 (cross-vendor) ·
VRAM floor 32 GB (32–40 GB fleet) · **full matrix is the default (~4 h)** ·
**weights deleted after each model** (`--keep-weights` to disable) ·
power metrics **best-effort** (null when the vendor tool lacks them) ·
per-machine standalone reports (no cross-run diff mode).

**No open items — ready to implement.**

---

## 12. Milestones

| # | Task | Notes |
|---|------|-------|
| 1 | Plan v3.1 review (this doc) | done |
| 2 | **Spike:** on one GPU, verify `vllm bench serve` JSON schema, MTP speculative-config names/flags for gpt-oss-20b + Qwen3-30B-A3B, `--kv-cache-dtype fp8` support, `--num-warmups` availability in v0.28.0 | Determines exact flags in models.yaml |
| 3 | `config/models.yaml` + `run_matrix.py` (server lifecycle, sweep, auto-skip) | |
| 4 | `telemetry.py` (nvidia-smi / rocm-smi / intel_gpu_top samplers) | |
| 5 | `report.py` (JSON + Markdown) | |
| 6 | `bench.sh` (detection, pull, run, exit codes) | |
| 7 | Smoke test: `--quick` on NVIDIA (this env or user's box) | |
| 8 | Full matrix run, NVIDIA | |
| 9 | Port checks: AMD, Intel (XPU FP8-KV/MTP auto-skip paths exercised) | |
| 10 | README + usage examples | |

---

*Draft v3.1 — 2026-09-02 (final: cyankiwi 27B, Qwen 35B cross-vendor, 32–40 GB fleet, full-matrix default, per-model weight deletion, best-effort power metrics, per-machine standalone reports; latest-gen models; measured on-disk sizes; MTP pre-check; NVFP4 cross-vendor exclusion)*