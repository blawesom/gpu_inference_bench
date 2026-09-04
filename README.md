# gpu_inference_bench

Platform-agnostic GPU inference benchmark. A single shell script (`bench.sh`)
detects the local GPU (NVIDIA / AMD / Intel), pulls the pinned vLLM image for
that vendor, runs a 4-model × optimization-config benchmark matrix against it,
and produces a standalone per-machine report.

- **`PLAN.md`** — the full design (model matrix, optimization configs,
  workload, report schema, script flow, error handling, milestones).
- **`spike.sh`** — milestone-2 verification script that validated the vLLM
  v0.28.0 flags / bench JSON schema / FP8-KV setup on a real target.

## How it works

```
host:  bench.sh
  1. preflight   docker present, vendor detected, disk gates (shared-FS aware)
  2. detect      nvidia-smi / /dev/kfd / xpu-smi / xe module → vendor
  3. pull        pinned vLLM image (vllm-openai[-rocm|-xpu]:v0.28.0)
  4. docker run  --entrypoint bash <vendor device flags>
        │
container: container/entrypoint.sh → container/run_matrix.py
  5. select GPU  in-container (largest VRAM, or --gpu-index)
  6. for each model (download → test; weights kept in cache):
       for each config:
         vllm serve <model> <config flags>  →  wait /health
           failure → parse log → cell skipped:<reason>, continue
         for C in 1 4 8 16:
           vllm bench serve --max-concurrency C  (1 Hz GPU telemetry)
  7. report.py   → report.json + report.md
```

The benchmark engine is `vllm bench serve` (built into the vLLM image) — no
custom HTTP client. Our code is a thin orchestrator: server lifecycle,
concurrency sweep, GPU telemetry sampling, aggregation, report rendering.

## Requirements

- Linux x86_64 host with Docker (images are amd64; ARM out of scope)
- A 32–40 GB VRAM GPU (the model matrix is sized to fit 32 GB; 40 GB just
  gains KV headroom)
- Per-vendor stack (the image carries the runtime, the host needs the driver):
  - **NVIDIA**: driver + `nvidia-container-toolkit`
  - **AMD**: ROCm kernel driver (`amdgpu` with KFD)
  - **Intel**: `xe` kernel module (Arc A/B dGPUs, e.g. Arc B70) on the host;
    the Level Zero runtime ships in the vLLM XPU image. `xpu-smi` is optional
    (host auto-detect only — `xe` module or lspci suffices)
- Disk: ~100 GB free on the first run (image ~25–35 GB + all model weights ~80 GB
  kept for re-runs).  With `--delete-weights` only ~65 GB is needed (one model at a
  time). `bench.sh` gates this automatically (`--force` to override; `--cache-dir`
  to put weights on a bigger volume).
- Network: Docker Hub + Hugging Face (all default models are ungated)

## Usage

```bash
./bench.sh                     # full matrix, auto-detect GPU (~3.5–4.5 h)
./bench.sh --quick             # smoke test: M1 only, baseline+kv-fp8, C=1,8
./bench.sh --models M2,M4      # subset of the matrix
./bench.sh --configs baseline  # subset of configs
./bench.sh --concurrency 1,8,16
./bench.sh --gpu-index 1       # force a specific physical GPU
./bench.sh --vendor amd        # override vendor detection
./bench.sh --image vllm/vllm-openai:v0.28.0   # override the image
./bench.sh --cache-dir /big/disk/hf           # HF weights cache location
./bench.sh --results /tmp/out                # output root
./bench.sh --delete-weights                   # delete weights after each model (old behavior)
./bench.sh --clean [M1,M2,...]                # remove cached weights, then exit
./bench.sh --start-timeout 1200             # server health-wait budget (s)
./bench.sh --validate --models M3,M4        # preflight VRAM-fit check (static+live probe)
./bench.sh --dry-run                        # print the docker command, stop
./bench.sh --force                          # skip disk-space gates
```

Environment overrides: `VLLM_VERSION`, `GPU_VENDOR`, `HF_CACHE_HOST`,
`RESULTS_DIR`, `SERVER_START_TIMEOUT`.

## Preflight validation (`--validate`)

`--validate` runs `container/validate_fit.py` instead of the benchmark matrix.
For each (model, config) cell it does two things:

1. **Static estimate** — no GPU required. Fetches `config.json` + weight-file
   sizes from HuggingFace and computes weight bytes, per-token KV cost
   (hybrid GDN aware: only the 1/4 full-attention layers carry a KV cache,
   the rest use a fixed-size linear-attention state), workload KV demand at
   C=16, and an overhead budget (CUDA graph + profiling + runtime). Verdict:
   **FIT / TIGHT / NO-FIT**. The overhead is auto-calibrated from the
   measured `mem_peak_gb` of the most recent local `report.json` if present,
   otherwise defaults to 6 GiB.
2. **Live probe** (when a GPU is available) — boots the exact `vllm serve`
   command for the cell, waits for `/health` or a startup failure, then parses
   vLLM's own sizing lines (`Model loading took`, `Estimated CUDA graph
   memory`, `Available KV cache memory`, `GPU KV cache size`) for the
   **definitive** verdict, including the exact error line on OOM/unsupported.

Output: `results/validate_<ts>_<gpu>/{fit_report.json,fit_report.md,
probe_<model>_<config>.log}`. The validator never runs the concurrency sweep
and leaves weights in the HF cache (the real run reuses them).

> **Why:** vLLM v0.28.0 counts CUDA-graph memory against
> `--gpu-memory-utilization`, so a checkpoint that "fits" on paper can still
> fail with `No available memory for the cache blocks`. `--validate` catches
> this before committing to a ~4 h matrix run — essential for the tight cells
> (M3 27 B dense, M4 35 B MoE on a 32 GB card).

## Outputs

Every run writes to its own directory — repeated runs never collide:

```
results/<YYYYMMDD-HHMMSS>_<gpu-model>/     # e.g. results/20260902-160512_nvidia-geforce-rtx-4090/
├── report.md          # human-readable: env block, model summary, per-model tables
├── report.json        # machine-readable: schema_version, run_id, environment,
│                      #   workload, models, metadata, rows (per cell × C)
├── environment.json   # GPU, driver, stack, OS, kernel, CPU, RAM, docker,
│                      #   image + ID, vLLM version
├── cells.json         # manifest: workload, common server settings, models, cells
├── server_<model>_<config>.log            # raw vLLM server log per cell
├── bench_<model>_<config>_<C>.json        # raw vllm bench serve output per level
└── telemetry_<model>_<config>_<C>.json    # 1 Hz GPU mem/util/power per level
results/.latest       # basename of the most recent run dir
```

The run-id is created *in-container* right after GPU selection, so the GPU
model name in the slug is the card actually benchmarked.

### Metrics

Per (model, config, concurrency) the report carries, from `vllm bench serve`:
TTFT / TPOT / ITL at p50/p90/p99, request & output token throughput,
completed/failed counts, wall time — plus our 1 Hz GPU telemetry (memory
peak/avg, utilization avg/peak, power avg/peak where the vendor tool exposes
it; otherwise `n/a`).

### Workload

Identical across all models and vendors for comparability: random synthetic
dataset, 512 input / 256 output tokens, 50 prompts per concurrency level,
2 warmups (excluded from metrics), seed 42, temperature 0, `--ignore-eos`,
concurrency sweep 1/4/8/16. All overridable in `config/models.yaml`.

## Model matrix

All four models fit the 32 GB floor (weights + KV cache + overhead), which is
what makes cross-machine results comparable:

| #   | Slot            | Model                              | Format      | On disk |
|-----|-----------------|------------------------------------|-------------|---------|
| M1  | dense ~9 B      | `Qwen/Qwen3.5-9B`                  | BF16        | 19.3 GB |
| M2  | MoE small       | `openai/gpt-oss-20b`               | MXFP4 native| 13.8 GB |
| M3  | dense 27 B      | `cyankiwi/Qwen3.8-27B-AWQ-INT4`    | 4-bit CT    | 21.0 GB |
| M4  | MoE 35 B / 3 B active | `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` | 4-bit CT  | 24.5 GB |

> **M4 checkpoint note:** the original `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` is
> not supported by the vLLM ROCm backend (the server fails to load the
> GPTQ attention/expert weights). The same base model in AWQ-4-bit
> (compressed-tensors, the quant family M3 already uses on ROCm) is the
> cross-vendor default; the GPTQ id remains in `config/models.yaml` as a
> commented NVIDIA-only alternative.
>
> **M3/M4 32 GB fit:** both are model-sized near the top of the 32 GB
> budget, so they carry model-level overrides in `config/models.yaml` —
> `gpu_memory_utilization: 0.95`, `max_model_len: 2048` (workload needs
> 512+256=768), and `--max-num-batched-tokens 2048` — because vLLM v0.28.0
> counts CUDA-graph memory against the utilization budget. Run `--validate`
> first; M4 is the tightest cell.

Per-model optimization configs (one server process each):

| Model | Configs |
|-------|---------|
| M1 (dense 9 B) | `baseline` · `kv-fp8` (`--kv-cache-dtype fp8`) · `long-context` (32 k) |
| M2 (MoE)       | `baseline` · `kv-fp8` |
| M3 (dense 27 B)| `baseline` · `kv-fp8` (long-context removed: 32 k ctx can't fit on 32 GB) |
| M4 (MoE 35 B)  | `baseline` · `kv-fp8` |

**Auto-skip:** if a backend rejects a config (e.g. FP8 KV cache unsupported
on some vendor/stack combos), the server fails to start, the orchestrator
marks the cell `skipped:<reason>` and continues — the report shows the full
matrix with pass/skip. A skip is *not* a failure (exit code 0); a genuine
failure (download error, >5 % failed requests, unexpected crash) exits 1.
The report is written either way.

## Storage behavior

Weights are **kept in the HF cache** after each model finishes, so re-runs
skip the ~20–25 GB re-download (speeds up iterative runs / A-B testing).
Peak disk ≈ image + **all** model weights + logs (~100 GB on the first run).

To free disk after a benchmark is done, use **`./clean.sh`**:

```bash
./clean.sh                    # remove ALL cached weights
./clean.sh M3,M4              # remove only M3 + M4 (by key from models.yaml)
./clean.sh "Qwen/Qwen3.5-9B"  # remove by HuggingFace repo ID
./clean.sh --dry-run          # preview what would be removed
```

`--delete-weights` (on `bench.sh`) restores the old per-model deletion (peak
disk back to ~40 GB) — useful when space is very tight.  `--keep-weights`
remains as a **no-op** for backward compatibility.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not detect GPU vendor` | Install the vendor stack (`nvidia-container-toolkit` / AMD KFD / Intel `xe` module), or pass `--vendor` |
| `no XPU device visible in the container` | Intel: check the `xe` module is loaded (`lsmod \| grep xe`) and `/dev/dri` exists; `bench.sh --dry-run` should show `-v /dev/dri:/dev/dri` in the docker command (oneCCL opens it as a directory). Inside the container, `zeinfo` must list the card |
| `no NVIDIA GPU found` | Check `nvidia-smi -L`; pass `--gpu-index` to pick a card |
| disk gate aborts | Weights are kept by default (whole set ≈ 85 GB). Use `--delete-weights` to restore the old one-model-at-a-time footprint, `--cache-dir <bigger volume>`, `./clean.sh` to free space, or `--force` |
| need to free weight disk | `./clean.sh` (all) or `./clean.sh M1,M2` (specific models) |
| cell `skipped:oom` | Model + graph + profiling exceed the util budget (vLLM v0.28 counts CUDA-graph memory); run `--validate` to see the exact shortfall, raise the model's `gpu_memory_utilization`, lower `max_model_len` / `max-num-batched-tokens`, or add `enforce-eager` as a last resort |
| cell `skipped:kv-fp8-unsupported` | Expected on backends without FP8 KV in v0.28.0; not a failure |
| cell `skipped:unsupported:<…>` | Backend rejected the model/quant format (e.g. **GPTQ on ROCm** — see `server_<model>_<config>.log` for the exact weight line). M4 defaults to the AWQ variant for this reason; `--validate` surfaces the exact error |
| model download fails | Gated repo → set `HF_TOKEN`; network → check HF access |
| server never healthy | `server_<model>_<config>.log` in the run dir; raise `--start-timeout` for big first loads |
| stale container from a crash | `docker rm -f gpu-bench` (bench.sh also does this on start) |

## Known limitations

- Single GPU per run (the largest-VRAM card, or `--gpu-index`).
- No speculative decoding: MTP is unsupported for every matrix model in
  vLLM v0.28.0 (verified in the spike); ngram is not a meaningful axis on the
  random workload.
- `--enable-prefix-caching` is deliberately off (it distorts synthetic-token
  throughput).
- Power draw is best-effort: `null` ("n/a") where the vendor tool lacks it.
- Reports are per-machine by design; there is no cross-run diff mode.