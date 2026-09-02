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
  2. detect      nvidia-smi / /dev/kfd / xpu-smi → vendor
  3. pull        pinned vLLM image (vllm-openai[-rocm|-xpu]:v0.28.0)
  4. docker run  --entrypoint bash <vendor device flags>
        │
container: container/entrypoint.sh → container/run_matrix.py
  5. select GPU  in-container (largest VRAM, or --gpu-index)
  6. for each model (download → test → delete weights):
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
  - **Intel**: XPU stack (`xpu-smi` / Level Zero)
- Disk: ~65 GB free (image ~25–35 GB + one model at a time ~25 GB).
  `bench.sh` gates this automatically (`--force` to override; `--cache-dir`
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
./bench.sh --keep-weights                   # don't delete weights per model
./bench.sh --start-timeout 1200             # server health-wait budget (s)
./bench.sh --dry-run                        # print the docker command, stop
./bench.sh --force                          # skip disk-space gates
```

Environment overrides: `VLLM_VERSION`, `GPU_VENDOR`, `HF_CACHE_HOST`,
`RESULTS_DIR`, `SERVER_START_TIMEOUT`.

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
| M3  | dense 27 B      | `cyankiwi/Qwen3.8-27B-AWQ-INT4`    | AWQ 4-bit   | 21.0 GB |
| M4  | MoE 35 B / 3 B active | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | GPTQ 4-bit | 24.4 GB |

Per-model optimization configs (one server process each):

| Model | Configs |
|-------|---------|
| M1, M3 (dense)  | `baseline` · `kv-fp8` (`--kv-cache-dtype fp8`) · `long-context` (32 k context) |
| M2, M4 (MoE)    | `baseline` · `kv-fp8` |

**Auto-skip:** if a backend rejects a config (e.g. FP8 KV cache unsupported
on some vendor/stack combos), the server fails to start, the orchestrator
marks the cell `skipped:<reason>` and continues — the report shows the full
matrix with pass/skip. A skip is *not* a failure (exit code 0); a genuine
failure (download error, >5 % failed requests, unexpected crash) exits 1.
The report is written either way.

## Storage behavior

Test machines are expected to have limited disk, so weights are **deleted
after each model** finishes (all its cells done, ok or failed). Peak disk ≈
image + one model + logs. `--keep-weights` disables deletion (debugging, or
re-running a single model without re-downloading).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not detect GPU vendor` | Install the vendor stack (`nvidia-container-toolkit` / AMD KFD / Intel XPU), or pass `--vendor` |
| `no NVIDIA GPU found` | Check `nvidia-smi -L`; pass `--gpu-index` to pick a card |
| disk gate aborts | Free space, `--cache-dir <bigger volume>`, or `--force` |
| cell `skipped:oom` | Model + KV too large for the card at 0.90 GPU util; try a smaller model or fewer concurrency levels |
| cell `skipped:kv-fp8-unsupported` | Expected on backends without FP8 KV in v0.28.0; not a failure |
| cell `skipped:unsupported:<…>` | Read the reason token, then `server_<model>_<config>.log` |
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