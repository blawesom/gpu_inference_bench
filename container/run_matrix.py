#!/usr/bin/env python3
"""run_matrix.py — orchestrate the GPU inference benchmark matrix.

Runs INSIDE the vLLM container. For each (model, config) cell:
  1. Download model weights to /hf-cache (idempotent; skipped if present).
  2. Start `vllm serve` with the config's flags; wait for /health.
  3. On startup failure, parse the server log → mark cell skipped:<reason>.
  4. On success, run the concurrency sweep (C=1,4,8,16), calling
     `vllm bench serve` per level with a 1 Hz GPU telemetry sample in flight.
  5. Kill the server; delete the model weights (unless --keep-weights).

Output (in /results):
  server_<model>_<config>.log          raw vLLM server log per cell
  bench_<model>_<config>_<C>.json      raw `vllm bench serve` output
  telemetry_<model>_<config>_<C>.json  1 Hz GPU samples for that level
  cells.json                           manifest consumed by report.py

Usage (from entrypoint.sh):
  python3 /bench/run_matrix.py \
      --config /bench/config/models.yaml \
      --results /results \
      --vendor amd [--models M1,M2] [--keep-weights] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML ships in the vllm image
    sys.exit("run_matrix.py: PyYAML is required (pip install pyyaml)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from telemetry import TelemetrySampler
except ImportError:
    TelemetrySampler = None  # telemetry optional (per-cell GPU metrics → null)

HF_CACHE = os.environ.get("HF_HOME", "/hf-cache")
HEALTH_PATH = "/health"
DEFAULT_START_TIMEOUT = int(os.environ.get("SERVER_START_TIMEOUT", "900"))


def _run_cmd(cmd: list[str], env: dict | None = None, timeout: float = 20.0) -> str | None:
    """Run a command, return stdout or None on any failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        if r.returncode == 0:
            return r.stdout
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def select_gpu(vendor: str, forced_idx: int | None = None) -> int:
    """Select a GPU inside the container and set the visibility env var.

    Returns the index to pass to the telemetry sampler:
      * AMD: the physical index (vllm sees it as its device 0 after
        HIP_VISIBLE_DEVICES; the sampler reads the *unfiltered* rocm-smi
        table, so it needs the physical index to pick the right row).
      * NVIDIA: 0 (bench.sh passes --gpus device=N, so only one GPU is
        visible in the container and it is device 0).
      * Intel: 0.
    """
    vendor = (vendor or "auto").lower()
    if vendor == "amd":
        env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
        out = _run_cmd(["rocm-smi", "--showmeminfo", "vram"], env=env, timeout=30.0)
        totals: dict[int, int] = {}
        for m in re.finditer(r"GPU\[(\d+)\].*?VRAM Total Memory \(B\): (\d+)",
                             out or ""):
            totals[int(m.group(1))] = int(m.group(2))
        if not totals:
            raise SystemExit("ERROR: no AMD GPU visible via rocm-smi")
        idx = forced_idx if forced_idx is not None else max(
            totals, key=lambda k: totals[k])
        if idx not in totals:
            raise SystemExit(f"ERROR: AMD GPU index {idx} not present "
                             f"(available: {sorted(totals)})")
        os.environ["HIP_VISIBLE_DEVICES"] = str(idx)
        print(f"[gpu-select] AMD physical idx {idx} "
              f"({totals[idx] // (1024 ** 3)} GB) → HIP_VISIBLE_DEVICES={idx}")
        return idx
    if vendor == "nvidia":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[gpu-select] NVIDIA: in-container device 0 "
              "(single GPU via --gpus device=N)")
        return 0
    if vendor == "intel":
        os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero/0")
        print("[gpu-select] Intel: level_zero/0")
        return 0
    raise SystemExit(f"ERROR: unsupported GPU_VENDOR {vendor!r} "
                     f"(expect amd|nvidia|intel)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU inference benchmark orchestrator")
    p.add_argument("--config", default="/bench/config/models.yaml")
    p.add_argument("--results", default="/results")
    p.add_argument("--vendor", default=os.environ.get("GPU_VENDOR", "auto"))
    p.add_argument("--gpu-index", type=int, default=None,
                   help="physical GPU index (overrides auto-pick by VRAM)")
    p.add_argument("--models", default=None, help="comma list, e.g. M1,M2")
    p.add_argument("--configs", default=None, help="comma list, e.g. baseline,kv-fp8")
    p.add_argument("--concurrency", default=None, help="comma list, e.g. 1,8,16")
    p.add_argument("--keep-weights", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="M1 only, baseline+kv-fp8, concurrency 1,8")
    p.add_argument("--start-timeout", type=int, default=DEFAULT_START_TIMEOUT)
    p.add_argument("--dry-run", action="store_true",
                   help="print planned server/bench commands, run nothing")
    return p.parse_args()


# ── Command builders ─────────────────────────────────────────────────────────
def build_server_cmd(model: dict, cfg: dict, common: dict) -> list[str]:
    """Assemble the `vllm serve ...` argv for one (model, config) cell.

    Precedence: common < model < config (e.g. long-context overrides
    max-model-len).
    """
    max_len = str(model["max_model_len"])
    extra: list[str] = []
    for flag, val in (cfg.get("flags") or {}).items():
        if flag == "max-model-len":
            max_len = "true" if val is True else str(val)
        elif val is True:
            extra.append(f"--{flag}")
        else:
            extra.extend([f"--{flag}", str(val)])

    cmd = ["vllm", "serve", model["id"],
           "--host", str(common.get("host", "0.0.0.0")),
           "--port", str(common.get("port", 8000)),
           "--max-model-len", max_len,
           "--gpu-memory-utilization", str(common.get("gpu_memory_utilization", 0.90))]
    if common.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    return cmd + extra


def build_bench_cmd(model: dict, workload: dict, concurrency: int,
                    out_file: str) -> list[str]:
    """Assemble the `vllm bench serve ...` argv for one concurrency level."""
    return [
        "vllm", "bench", "serve",
        "--host", "127.0.0.1", "--port", str(workload.get("port", 8000)),
        "--backend", workload.get("backend", "openai-chat"),
        "--endpoint", workload.get("endpoint", "/v1/chat/completions"),
        "--model", model["id"],
        "--dataset-name", "random",
        "--random-input-len", str(workload.get("random_input_len", 512)),
        "--random-output-len", str(workload.get("random_output_len", 256)),
        "--num-prompts", str(workload.get("num_prompts", 50)),
        "--max-concurrency", str(concurrency),
        "--num-warmups", str(workload.get("num_warmups", 2)),
        "--seed", str(workload.get("seed", 42)),
        "--temperature", str(workload.get("temperature", 0)),
        "--ignore-eos",
        "--percentile-metrics", workload.get("percentile_metrics", "ttft,tpot,itl"),
        "--metric-percentiles", workload.get("metric_percentiles", "50,90,99"),
        "--save-result",
        "--result-filename", out_file,
    ]


# ── Server lifecycle ─────────────────────────────────────────────────────────
def wait_health(port: int, timeout: float) -> bool:
    url = f"http://127.0.0.1:{port}{HEALTH_PATH}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlrequest.urlopen(url, timeout=5) as r:
                if 200 <= r.status < 300:
                    return True
        except (URLError, OSError, ValueError):
            pass
        time.sleep(5)
    return False


def start_server(cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    proc._log_fh = log_fh  # type: ignore[attr-defined]  # keep handle alive
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
    except (ProcessLookupError, OSError):
        pass
    finally:
        try:
            proc._log_fh.close()  # type: ignore[attr-defined]
        except Exception:
            pass


# ── Skip-reason detection (log patterns) ─────────────────────────────────────
OOM_PATTERNS = re.compile(
    r"out of memory|\bOOM\b|no available memory for the cache blocks|"
    r"cannot allocate|HIP(?:_|\s)?ERROR_OUT_OF_MEMORY|MemoryAlloc", re.I)
UNSUPPORTED_PATTERNS = re.compile(
    r"not supported|not implemented|no kernel|unsupported|"
    r"no (?:handler|support) for|could not be used|is not available", re.I)


def parse_skip_reason(log_path: Path) -> tuple[str, str]:
    """Return (status, reason) from a failed server log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return "failed", "no-server-log"
    if OOM_PATTERNS.search(text):
        return "skipped", "oom"
    if re.search(r"kv[ -]?cache.{0,40}(fp8|dtype)|fp8.{0,40}(not support|unavail)",
                 text, re.I):
        return "skipped", "kv-fp8-unsupported"
    m = UNSUPPORTED_PATTERNS.search(text)
    if m:
        line = text[max(0, m.start() - 60):m.end() + 40].replace("\n", " ")
        token = re.sub(r"[^a-z0-9]+", "-", line.lower()).strip("-")[:40]
        return "skipped", f"unsupported:{token}"
    return "failed", "engine-startup"


# ── Weight lifecycle ─────────────────────────────────────────────────────────
def weights_dir(model_id: str) -> Path:
    return Path(HF_CACHE) / "hub" / ("models--" + model_id.replace("/", "--"))


def download_weights(model_id: str) -> None:
    d = weights_dir(model_id)
    if d.exists() and any(d.iterdir()):
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_id, cache_dir=HF_CACHE)


def delete_weights(model_id: str) -> bool:
    d = weights_dir(model_id)
    try:
        if d.exists():
            shutil.rmtree(d)
            return True
    except OSError as e:
        print(f"  WARN: could not delete weights for {model_id}: {e}",
              file=sys.stderr)
    return False


# ── Orchestration ────────────────────────────────────────────────────────────
def run_cell(model_key: str, model: dict, cfg: dict, workload: dict, common: dict,
             results: Path, vendor: str, gpu_index: int | None, concurrencies: list[int],
             start_timeout: int, dry_run: bool) -> dict:
    tag = f"{model_key}_{cfg['name']}"
    server_log = results / f"server_{tag}.log"
    port = int(common.get("port", 8000))
    s_cmd = build_server_cmd(model, cfg, common)
    print(f"[cell {tag}] server: {' '.join(s_cmd)}")

    cell: dict = {
        "model": model_key, "model_id": model["id"], "config": cfg["name"],
        "status": "failed", "reason": None,
        "server_flags": {
            "max_model_len": model["max_model_len"],
            "gpu_memory_utilization": common.get("gpu_memory_utilization", 0.90),
            **{f.replace("-", "_"): v for f, v in (cfg.get("flags") or {}).items()},
        },
        "server_log": server_log.name,
        "concurrency_results": {},
    }
    if dry_run:
        for c in concurrencies:
            b_cmd = build_bench_cmd(model, workload, c, f"bench_{tag}_{c}.json")
            print(f"[cell {tag}]   bench C={c}: {' '.join(b_cmd)}")
        cell["status"] = "dry-run"
        return cell

    server: subprocess.Popen | None = None
    try:
        server = start_server(s_cmd, server_log)
        if not wait_health(port, start_timeout):
            cell["status"], cell["reason"] = parse_skip_reason(server_log)
            print(f"[cell {tag}] server FAILED → {cell['status']}:{cell['reason']}")
            return cell
        cell["status"] = "ok"
        print(f"[cell {tag}] server healthy → sweeping C={concurrencies}")
    except Exception as e:  # defensive: never lose a cell to an exception
        cell["status"], cell["reason"] = "failed", f"orchestration:{e}"
        return cell

    sampler = (TelemetrySampler(vendor, gpu_index) if TelemetrySampler else None)
    try:
        for c in concurrencies:
            bench_out = results / f"bench_{tag}_{c}.json"
            b_cmd = build_bench_cmd(model, workload, c, str(bench_out))
            print(f"[cell {tag}]   bench C={c} ...")
            samples = []
            if sampler:
                samples = sampler.start()
            t0 = time.time()
            proc = subprocess.run(b_cmd, capture_output=True, text=True)
            wall = time.time() - t0
            if sampler:
                samples = sampler.stop()
            telem_out = results / f"telemetry_{tag}_{c}.json"
            telem = sampler.aggregate(samples) if (sampler and samples) else None
            if telem_out and telem is not None:
                telem_out.write_text(json.dumps(telem, indent=2))

            level: dict = {
                "bench_json": bench_out.name,
                "telemetry_json": (telem_out.name if telem is not None else None),
                "status": "ok", "reason": None, "wall_time_s": round(wall, 1),
            }
            if proc.returncode != 0 or not bench_out.exists():
                level["status"] = "failed"
                level["reason"] = (proc.stderr or proc.stdout or "")[-400:]
            cell["concurrency_results"][str(c)] = level
    finally:
        stop_server(server)
    return cell


def run_model(model_key: str, model: dict, cfgs: list[dict], workload: dict,
              common: dict, results: Path, vendor: str, gpu_index: int | None,
              concurrencies: list[int], start_timeout: int,
              keep_weights: bool, dry_run: bool) -> list[dict]:
    cells = []
    if not dry_run:
        print(f"[model {model_key}] downloading {model['id']} ...")
        try:
            download_weights(model["id"])
        except Exception as e:
            print(f"[model {model_key}] download FAILED: {e}", file=sys.stderr)
            for cfg in cfgs:
                cells.append({"model": model_key, "model_id": model["id"],
                              "config": cfg["name"], "status": "failed",
                              "reason": f"download:{e}", "server_log": None,
                              "concurrency_results": {}})
            return cells
    for cfg in cfgs:
        cells.append(run_cell(model_key, model, cfg, workload, common, results,
                              vendor, gpu_index, concurrencies, start_timeout, dry_run))
    if not dry_run and not keep_weights:
        removed = delete_weights(model["id"])
        for c in cells:
            c["weights_removed"] = removed
        print(f"[model {model_key}] weights removed: {removed}")
    return cells


def main() -> int:
    args = parse_args()
    doc = yaml.safe_load(Path(args.config).read_text())
    workload, common, models = doc["workload"], doc["common_server"], doc["models"]
    configs = doc.get("configs", {})
    configs_by_name = {name: {**cfg, "name": name} for name, cfg in configs.items()}

    if args.quick:
        model_keys, names, concurrencies = ["M1"], ["baseline", "kv-fp8"], [1, 8]
    else:
        model_keys, names, concurrencies = (
            list(models.keys()), None,
            list(workload.get("concurrency_levels", [1, 4, 8, 16])))
    if args.models:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.concurrency:
        concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    if args.configs:
        names = [c.strip() for c in args.configs.split(",") if c.strip()]

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    gpu_index: int | None = None
    if not args.dry_run:
        gpu_index = select_gpu(args.vendor, args.gpu_index)

    all_cells: list[dict] = []
    for mk in model_keys:
        if mk not in models:
            print(f"WARN: unknown model {mk}, skipping", file=sys.stderr)
            continue
        model = models[mk]
        sel = names or model.get("configs", list(configs_by_name.keys()))
        cfgs = [configs_by_name[n] for n in sel if n in configs_by_name]
        all_cells.extend(run_model(mk, model, cfgs, workload, common, results,
                                   args.vendor, gpu_index, concurrencies,
                                   args.start_timeout, args.keep_weights,
                                   args.dry_run))

    (results / "cells.json").write_text(json.dumps(all_cells, indent=2))
    n_ok = sum(1 for c in all_cells if c["status"] == "ok")
    n_skip = sum(1 for c in all_cells if c["status"].startswith("skipped"))
    n_fail = sum(1 for c in all_cells if c["status"] == "failed")
    print(f"\n[run_matrix] {n_ok} ok, {n_skip} skipped, {n_fail} failed "
          f"of {len(all_cells)} cells → {results / 'cells.json'}")
    return 1 if n_fail and not args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())