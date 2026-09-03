#!/usr/bin/env python3
"""validate_fit.py — pre-flight validation of model VRAM fit on the target GPU.

Two modes (chosen automatically when a GPU is present):

  1. Static estimate — no GPU required. Fetches config.json + file sizes from
     HuggingFace, computes weight bytes, per-token KV cost, workload KV demand,
     and applies an overhead budget to produce a FIT / TIGHT / NO-FIT verdict.

  2. Live probe — boots the vLLM server with the cell's exact flags, waits for
     /health or OOM, parses vLLM's own memory-sizing log lines, and reports
     the decisive verdict with actual numbers.

Usage (inside the vLLM container):
  python3 /bench/container/validate_fit.py \\
      --config /bench/config/models.yaml \\
      --results /results \\
      --vendor amd [--models M1,M2,M3,M4] [--probe] [--start-timeout 600]

Usage (host-side, static-only, no GPU):
  python3 /bench/container/validate_fit.py \\
      --config /bench/config/models.yaml \\
      --results /results \\
      --vendor amd --static-only --vram-gb 31.9 [--models M3,M4]
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
from datetime import datetime
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

try:
    import yaml
except ImportError:
    sys.exit("validate_fit.py: PyYAML required (pip install pyyaml)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry import TelemetrySampler  # type: ignore

try:
    from run_matrix import select_gpu, gpu_name, build_server_cmd  # type: ignore
except ImportError:
    select_gpu, gpu_name, build_server_cmd = None, None, None  # static-only fallback

HF_CACHE = os.environ.get("HF_HOME", "/hf-cache")
HEALTH_PATH = "/health"
DEFAULT_START_TIMEOUT = int(os.environ.get("SERVER_START_TIMEOUT", "600"))
DEFAULT_OVERHEAD_GIB = 6.0  # CUDA graph + profiling + runtime on this stack


# ── HF API helpers (static estimate, no GPU needed) ───────────────────────────


def _get_json(url: str) -> dict | None:
    req = urlrequest.Request(url, headers={"User-Agent": "gpu-bench-validate"})
    try:
        with urlrequest.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _get_file(url: str) -> str | None:
    req = urlrequest.Request(url, headers={"User-Agent": "gpu-bench-validate"})
    try:
        with urlrequest.urlopen(req, timeout=30) as r:
            return r.read().decode(errors="replace")
    except Exception:
        return None


def weights_gb_static(repo_id: str) -> float | None:
    """Total weight-file bytes from HF API (safetensors / bin)."""
    info = _get_json(f"https://huggingface.co/api/models/{repo_id}?blobs=true")
    if not info:
        return None
    total = 0
    for sib in info.get("siblings", []):
        rfn = sib.get("rfilename", "")
        if rfn.endswith((".safetensors", ".bin")):
            total += sib.get("size", 0)
    return round(total / 1e9, 2) if total else None


def config_static(repo_id: str) -> dict | None:
    """config.json from HF (raw text, parsed)."""
    raw = _get_file(f"https://huggingface.co/{repo_id}/raw/main/config.json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _kv_bytes_per_token(config: dict, kv_dtype_bytes: int = 2,
                        is_fp8: bool = False) -> tuple[float, float]:
    """Compute KV cache footprint for a hybrid GDN model.

    Returns (full_attn_kv_per_token_bytes, gdn_state_per_request_bytes).

    Hybrid GDN architecture: `full_attention_interval` layers (default 4) have
    a full-attention KV cache; the remaining 3/4 use linear (GDN) attention
    with a fixed-size recurrent state per layer.

    For non-hybrid models the full-attn path is used for all layers.
    """
    tc = config.get("text_config", config)
    n_layers = tc.get("num_hidden_layers", 0)
    n_kv_heads = tc.get("num_key_value_heads", 0)
    head_dim = tc.get("head_dim", 0)
    full_interval = tc.get("full_attention_interval", 4)  # 4 → 1/4 layers have KV

    # Full-attention layers
    n_full = n_layers // full_interval if full_interval and n_layers >= full_interval else n_layers

    # GDN state per request: per-layer recurrent state ≈ v_heads × k_dim × v_dim × 2B
    lin_v = tc.get("linear_num_value_heads", 0)
    lin_k = tc.get("linear_key_head_dim", 0)
    lin_vd = tc.get("linear_value_head_dim", 0)
    n_linear = n_layers - n_full if n_layers > n_full else 0
    gdn_per_layer = lin_v * lin_k * lin_vd * 2  # fp16 recurrent state, bytes
    gdn_per_request = n_linear * gdn_per_layer

    bytes_per_token = 2 * n_full * n_kv_heads * head_dim * kv_dtype_bytes  # K+V per token
    return bytes_per_token, gdn_per_request


# ── Overhead calibration from a past run ──────────────────────────────────────


def _calibrate_overhead(results_dir: str | None, common: dict) -> float:
    """Auto-calibrate the overhead estimate from a past report.json.

    For each ok row with telemetry, overhead = (vram × util) − measured mem
    peak (the unmeasured remainder: CUDA graph + profiling transient +
    runtime). Returns the worst case across rows, or the default.
    """
    if not results_dir:
        return DEFAULT_OVERHEAD_GIB
    rpath = Path(results_dir)
    candidates = sorted(rpath.glob("report.json"))
    # per-run dirs: results/<run>/report.json — search two levels deep
    if not candidates:
        candidates = sorted(rpath.glob("*/report.json"))
    for report in reversed(candidates):
        try:
            data = json.loads(report.read_text())
            env = data.get("environment") or {}
            vram = env.get("vram_total_gb")
            util = (data.get("common_server") or common).get(
                "gpu_memory_utilization", 0.90)
            rows = data.get("rows", [])
            peaks = [row["telemetry"]["mem_peak_gb"] for row in rows
                     if row.get("status") == "ok"
                     and (row.get("telemetry") or {}).get("mem_peak_gb")]
            if vram and peaks:
                budget_gb = vram * util
                over = [round(budget_gb - p, 2) for p in peaks
                        if budget_gb - p > 0]
                if over:
                    return max(over)
        except Exception:
            continue
    return DEFAULT_OVERHEAD_GIB


# ── Static Estimate ──────────────────────────────────────────────────────────


def static_estimate(model: dict, cfg: dict, workload: dict,
                    vram_gb: float, gpu_mem_util: float,
                    overhead_gib: float, vendor: str, is_fp8: bool) -> dict:
    """Compute a memory-fit estimate for one (model, config).

    Returns a dict with all intermediate numbers + the verdict.
    """
    repo_id = model["id"]
    kv_dtype = "fp8" if is_fp8 else "bf16"
    kv_bytes = 1 if is_fp8 else 2  # fp8=1 byte/token/head/dim, bf16=2

    # --- weights ---
    w_gb = model.get("weights_gb") or weights_gb_static(repo_id)
    if w_gb is None:
        return {"verdict": "NO-FIT", "reason": "could not determine weight size",
                "repo": repo_id}
    w_gib = w_gb * 1024 / 1024  # rough GB → GiB

    # --- budget ---
    budget_gb = vram_gb * gpu_mem_util
    budget_gib = budget_gb * 1024 / 1024

    # --- config.json ---
    cfg_json = config_static(repo_id)
    if not cfg_json:
        return {"verdict": "TIGHT", "reason": "could not fetch config.json",
                "weights_gb": w_gb}

    tc = cfg_json.get("text_config", cfg_json)
    n_layers = tc.get("num_hidden_layers", 0)
    n_kv_heads = tc.get("num_key_value_heads", 0)
    head_dim = tc.get("head_dim", 0)
    full_interval = tc.get("full_attention_interval", 4)

    n_full_attn = n_layers // full_interval if full_interval and n_layers >= full_interval else n_layers

    # --- KV cost ---
    kv_per_token, gdn_per_request = _kv_bytes_per_token(cfg_json, kv_dtype_bytes=kv_bytes, is_fp8=is_fp8)

    # --- workload demand ---
    input_len = workload.get("random_input_len", 512)
    output_len = workload.get("random_output_len", 256)
    max_tokens = input_len + output_len
    max_conc = max(workload.get("concurrency_levels", [16]))
    total_tokens = max_conc * max_tokens  # max tokens in flight
    total_kv_bytes = total_tokens * kv_per_token
    gdn_bytes = max_conc * gdn_per_request

    # --- pool ---
    pool_gib = budget_gib - w_gib - overhead_gib
    pool_tokens_fp8 = pool_gib * 1024 * 1024 * 1024 / kv_per_token if kv_per_token else 0

    # --- verdict ---
    if w_gib >= budget_gib:
        verdict = "NO-FIT"
        reason = "weights alone exceed budget"
    elif pool_gib <= 0:
        verdict = "NO-FIT"
        reason = f"budget - weights - overhead ({pool_gib:+.1f} GiB) → OOM at startup"
    else:
        pool_giB_actual = pool_gib
        gdn_giB_actual = gdn_bytes / 1024 / 1024 / 1024
        usable_pool = max(pool_giB_actual - gdn_giB_actual, 0)  # GDN state comes from pool
        if usable_pool > 0:
            usable_tokens = usable_pool * 1024 * 1024 * 1024 / kv_per_token
            if usable_tokens >= total_tokens:
                verdict = "FIT"
                reason = f"pool ≈ {usable_tokens/1000:.0f}k tokens, demand = {total_tokens/1000:.0f}k"
            else:
                pct = usable_tokens / total_tokens * 100
                verdict = "TIGHT"
                reason = f"pool ≈ {usable_tokens/1000:.0f}k tokens ({pct:.0f}% of demand); will queue at C={max_conc}"
        else:
            verdict = "TIGHT"
            reason = "pool fully consumed by GDN state; no KV blocks"

    return {
        "verdict": verdict, "reason": reason,
        "repo": repo_id,
        "weights_gb": w_gb, "weights_gib": round(w_gib, 2),
        "budget_gb": round(budget_gb, 2), "budget_gib": round(budget_gib, 2),
        "overhead_gib": round(overhead_gib, 2),
        "n_layers": n_layers, "n_full_attn_layers": n_full_attn,
        "kv_per_token_bytes": kv_per_token, "kv_per_token_kb": round(kv_per_token / 1024, 1),
        "gdn_per_request_bytes": gdn_per_request, "gdn_per_request_kb": round(gdn_per_request / 1024, 1),
        "total_tokens_flight": total_tokens, "total_tokens_kb": round(total_tokens / 1024, 1),
        "pool_gib": round(pool_gib, 2), "pool_tokens_approx": round(pool_tokens_fp8 / 1000, 0),
        "gpu_mem_util": gpu_mem_util, "kv_dtype": kv_dtype,
    }


# ── Live Probe ───────────────────────────────────────────────────────────────

PROBE_PATTERNS = re.compile(
    r"(Model loading took|Estimated CUDA graph memory|Available KV cache memory|"
    r"GPU KV cache size|GPU KV cache|number of GPU blocks|Maximum concurrency)",
    re.I,
)
OOM_PATTERNS = re.compile(
    r"out of memory|\bOOM\b|no available memory for the cache blocks|"
    r"cannot allocate|HIP(?:_|\s)?ERROR_OUT_OF_MEMORY|MemoryAlloc", re.I)
UNSUPPORTED_PATTERNS = re.compile(
    r"not supported|not implemented|no kernel|unsupported|"
    r"no (?:handler|support) for|could not be used|is not available", re.I)


def _parse_probe_log(log_path: Path) -> dict:
    """Parse the decisive memory/fit lines from a vLLM server log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}

    result: dict = {}
    m = re.search(r"Model loading took ([\d.]+) GiB memory", text)
    if m:
        result["model_load_gib"] = float(m.group(1))
    m = re.search(r"Estimated CUDA graph memory: ([\d.]+) GiB", text)
    if m:
        result["graph_gib"] = float(m.group(1))
    m = re.search(r"Available KV cache memory: (-?[\d.]+) GiB", text)
    if m:
        result["available_kv_gib"] = float(m.group(1))
    # KV cache size: v0.28 may say "GPU KV cache size: N tokens" (numbers are
    # comma-separated, e.g. "36,864 tokens") or "number of GPU blocks: K"
    # (each block = 16 tokens by default).
    m = re.search(r"(?:GPU KV cache size|KV cache size):\s*([\d,]+)\s*tokens?", text)
    if m:
        result["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    else:
        m = re.search(r"number of GPU blocks:\s*([\d,]+)", text)
        if m:
            result["kv_cache_tokens"] = int(m.group(1).replace(",", "")) * 16
    m = re.search(r"Maximum concurrency for\s*(\d+)\s*tokens per request:\s*([\d.]+)%", text)
    if m:
        result["max_conc_pct"] = float(m.group(2))

    if OOM_PATTERNS.search(text):
        result["status"] = "OOM"
    elif UNSUPPORTED_PATTERNS.search(text):
        result["status"] = "UNSUPPORTED"
        # Capture the 3-line context around the match
        m2 = UNSUPPORTED_PATTERNS.search(text)
        if m2:
            start = max(0, m2.start() - 30)
            end = min(len(text), m2.end() + 60)
            line_before = text.rfind("\n", 0, start)
            line_after = text.find("\n", end)
            result["unsupported_context"] = text[line_before:line_after].strip().replace("\n", " | ")
    else:
        result["status"] = "OK"

    # Also grab the log line where the server actually failed to start
    if result.get("status") in ("OOM", "UNSUPPORTED"):
        # Find the error line (first ValueError / RuntimeError / ERROR line)
        for line in text.splitlines():
            if re.search(r"ValueError|RuntimeError|ERROR", line):
                result["error_line"] = line.strip()
                break

    return result


def start_server(cmd: list[str], log_path: Path) -> subprocess.Popen | None:
    log_fh = open(log_path, "w")
    try:
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        proc._log_fh = log_fh
        return proc
    except Exception:
        log_fh.close()
        return None


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
            proc._log_fh.close()
        except Exception:
            pass


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


def probe_cell(model: dict, cfg: dict, workload: dict, common: dict,
               vendor: str, gpu_index: int | None, log_dir: Path,
               start_timeout: int = 300) -> dict:
    """Launch the server, wait for health (or OOM), parse log → verdict."""
    server_log = log_dir / (
        f"probe_{model.get('id','').replace('/','_')}_{cfg.get('name','')}.log")
    port = int(common.get("port", 8000))
    s_cmd = build_server_cmd(model, cfg, common)

    # Download weights (reuse cache)
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo_id=model["id"], cache_dir=HF_CACHE)
    except Exception as e:
        return {"verdict": "NO-FIT", "reason": f"download failed: {e}",
                "probe_cmd": s_cmd, "log": server_log.name}

    server = start_server(s_cmd, server_log)
    if not wait_health(port, start_timeout):
        parsed = _parse_probe_log(server_log)
        stop_server(server)
        parsed["probe_cmd"] = s_cmd
        parsed["log"] = server_log.name
        if parsed.get("status") == "OK":
            parsed["status"] = "TIMEOUT"
        return {"verdict": "NO-FIT" if parsed.get("status") == "OOM" else "UNSUPPORTED",
                "reason": parsed.get("status", "unknown"),
                "log_lines": parsed}
    parsed = _parse_probe_log(server_log)
    parsed["probe_cmd"] = s_cmd
    parsed["log"] = server_log.name
    parsed["healthy"] = True
    stop_server(server)

    # Determine verdict from vLLM's own numbers
    avail = parsed.get("available_kv_gib", None)
    if parsed.get("status") == "TIMEOUT":
        verdict, reason = "TIMEOUT", "server did not become healthy in time"
    elif avail is not None and avail < 0:
        verdict, reason = "NO-FIT", f"Available KV: {avail:.1f} GiB"
    elif parsed.get("status") == "UNSUPPORTED":
        verdict, reason = "UNSUPPORTED", parsed.get("unsupported_context", "") or \
            "quant format not supported on this backend"
    else:
        kv_tokens = parsed.get("kv_cache_tokens", 0)
        workload_tokens = max(workload.get("concurrency_levels", [16])) * \
            (workload.get("random_input_len", 512) + workload.get("random_output_len", 256))
        if kv_tokens >= workload_tokens:
            verdict, reason = "FIT", f"KV pool {kv_tokens:,} tokens ≥ demand {workload_tokens:,}"
        else:
            pct = kv_tokens / workload_tokens * 100 if workload_tokens else 0
            verdict, reason = "TIGHT", f"KV pool {kv_tokens:,} tokens ({pct:.0f}% of demand)"
    parsed["verdict"] = verdict
    parsed["reason"] = reason
    return parsed


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate model VRAM fit on target GPU")
    p.add_argument("--config", default="/bench/config/models.yaml",
                    help="path to models.yaml config")
    p.add_argument("--results", default="/results",
                    help="output root directory")
    p.add_argument("--vendor", default=os.environ.get("GPU_VENDOR", "auto"),
                    help="amd | nvidia | intel | auto (default)")
    p.add_argument("--gpu-index", type=int, default=None,
                    help="physical GPU index (overrides auto-pick by VRAM)")
    p.add_argument("--models", default=None, help="comma list, e.g. M3,M4")
    p.add_argument("--start-timeout", type=int, default=DEFAULT_START_TIMEOUT,
                    help="server health-wait budget in seconds")
    p.add_argument("--overhead-gib", type=float, default=None,
                    help="override overhead estimate (GiB); default: auto-calibrate from past run or 6.0")
    p.add_argument("--vram-gb", type=float, default=None,
                    help="manual VRAM in GB (static-only mode)")
    p.add_argument("--no-probe", action="store_true",
                    help="skip live server probe (static only)")
    p.add_argument("--quiet", action="store_true", help="suppress stdout, write files only")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    doc = yaml.safe_load(Path(args.config).read_text())
    workload, common, models = doc["workload"], doc["common_server"], doc["models"]
    configs_by_name = {n: {**c, "name": n} for n, c in doc.get("configs", {}).items()}

    if args.models:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_keys = list(models.keys())

    # Determine vram
    vram_gb = args.vram_gb
    gpu_idx = None
    vendor = (args.vendor or "auto").lower()
    if not vram_gb:
        if vendor == "amd":
            env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                                 capture_output=True, text=True, env=env, timeout=30).stdout
            m = re.search(r"VRAM Total Memory \(B\): (\d+)", out)
            if m:
                vram_gb = int(m.group(1)) / 1024**3
            else:
                vram_gb = 32.0  # fallback
        elif vendor == "nvidia":
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True)
            m = re.search(r"([\d.]+) MiB", out.stdout)
            if m:
                vram_gb = float(m.group(1)) / 1024.0
            else:
                vram_gb = 32.0
        else:
            vram_gb = 32.0

    # Auto-select GPU for probe mode
    probe_mode = True
    gpu_name_str = f"{vendor}-gpu"
    if args.no_probe or select_gpu is None:
        probe_mode = False
        if not args.quiet:
            print(f"[validate] static-only mode (no GPU selected, --no-probe)")
    else:
        gpu_idx = select_gpu(vendor, args.gpu_index)
        gpu_name_str = gpu_name(vendor, gpu_idx) or f"{vendor}-gpu"

    # Calibrate overhead from past run
    results = Path(args.results)
    overhead_gib = args.overhead_gib
    if overhead_gib is None:
        overhead_gib = _calibrate_overhead(str(results), common)
        if not args.quiet:
            print(f"[validate] overhead estimate: {overhead_gib:.1f} GiB "
                  f"(auto-calibrated from past run, or default)")

    # Budget (per-model utilization override applied below)
    common_util = common.get("gpu_memory_utilization", 0.90)

    # Create the run dir up-front so probe logs land inside it.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", (gpu_name_str or f"{vendor}-gpu").lower()).strip("-") or "gpu"
    run_dir, n = results / f"validate_{ts}_{slug}", 2
    while run_dir.exists():
        run_dir, n = results / f"validate_{ts}_{slug}-{n}", n + 1
    run_dir.mkdir(parents=True)

    rows: list[dict] = []
    all_pass = True

    for mk in model_keys:
        if mk not in models:
            if not args.quiet:
                print(f"[validate] unknown model {mk}, skipping")
            continue
        model = models[mk]
        util = model.get("gpu_memory_utilization", common_util)
        sel = model.get("configs", list(configs_by_name.keys()))
        for cfg_name in sel:
            if cfg_name not in configs_by_name:
                continue
            cfg = configs_by_name[cfg_name]
            is_fp8 = cfg.get("flags", {}).get("kv-cache-dtype") == "fp8"

            # --- Static estimate (uses model-level utilization) ---
            est = static_estimate(model, cfg, workload, vram_gb, util,
                                  overhead_gib, vendor, is_fp8)
            est["model"] = mk
            est["model_id"] = model["id"]
            est["config"] = cfg["name"]
            est["probe_run"] = probe_mode

            if probe_mode:
                # --- Live probe (definitive: vLLM's own sizing numbers) ---
                if not args.quiet:
                    print(f"[validate] probing {mk}/{cfg['name']} "
                          f"(budget {vram_gb*util:.1f} GB, w={est['weights_gb']} GB) ...",
                          end=" ", flush=True)
                probe = probe_cell(model, cfg, workload, common,
                                   vendor, gpu_idx, run_dir,
                                   start_timeout=args.start_timeout)
                est["probe"] = probe
                pverdict = probe.get("verdict")
                if pverdict in ("NO-FIT", "UNSUPPORTED", "TIMEOUT"):
                    est["verdict"] = pverdict
                    est["reason"] = probe.get("reason") or probe.get("status", "")
                    all_pass = False
                elif pverdict == "TIGHT":
                    est["verdict"] = "TIGHT"
                    est["reason"] = probe.get("reason", est.get("reason", ""))
                else:
                    est["verdict"] = "FIT"
                    est["reason"] = probe.get("reason", est.get("reason", ""))

                if not args.quiet:
                    print(f"probe → {pverdict} ({probe.get('reason', '')[:90]})")

            if not args.quiet:
                verdict_color = {"FIT": "\033[92m", "TIGHT": "\033[93m", "NO-FIT": "\033[91m",
                                 "UNSUPPORTED": "\033[91m"}.get(est["verdict"], "\033[0m")
                print(f"  {verdict_color}[{est['verdict']}]\033[0m "
                      f"w={est['weights_gb']}GB "
                      f"KV/k={est.get('kv_per_token_kb','?')}KB "
                      f"pool={est.get('pool_gib','?')}GiB "
                      f"→ {est.get('reason','')[:80]}")

            rows.append(est)

    # ── Write output ───────────────────────────────────────────────────────
    fit_report = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "generated_at": datetime.now().isoformat(),
        "vram_gb": round(vram_gb, 2),
        "gpu": slug,
        "overhead_gib_used": overhead_gib,
        "common_gpu_memory_utilization": common_util,
        "probe_mode": probe_mode,
        "rows": rows,
    }
    (run_dir / "fit_report.json").write_text(json.dumps(fit_report, indent=2) + "\n")

    # ── Markdown summary ───────────────────────────────────────────────────
    lines = [
        f"# GPU Inference Bench — Fit Validation",
        "",
        f"**{slug}** · Run: {run_dir.name} · {datetime.now().isoformat()}",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| VRAM (GB) | {vram_gb:.1f} |",
        f"| Common utilization | {common_util} (models may override) |",
        f"| Overhead estimate | {overhead_gib:.1f} GiB (graph + profiling + runtime) |",
        f"| Probe mode | {'live (definitive)' if probe_mode else 'static-only (estimate)'} |",
        "",
        "| Model | Config | Verdict | Weights GB | Util | KV/token | Pool est. GiB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        util = r.get("gpu_mem_util", common_util)
        lines.append(f"| {r.get('model','')} | {r.get('config','')} "
                      f"| **{r['verdict']}** | {r.get('weights_gb','?')} "
                      f"| {util} "
                      f"| {r.get('kv_per_token_kb','?')}KB "
                      f"| {r.get('pool_gib','?')} |")
        if r.get("reason"):
            lines.append(f"| | | ↳ {r['reason'][:140]} | | | | |")
    lines.append("")
    lines.append("## Verdict Legend")
    lines.append("- **FIT**: weights + graph + profiling leave enough KV pool for full C=16 workload")
    lines.append("- **TIGHT**: KV pool will be smaller than demand; server starts but queues at C=16")
    lines.append("- **NO-FIT**: startup OOM; server cannot initialize")
    lines.append("- **UNSUPPORTED**: model/quant format not supported by this vLLM backend")
    lines.append("")

    n_fit = sum(1 for r in rows if r["verdict"] == "FIT")
    n_tight = sum(1 for r in rows if r["verdict"] == "TIGHT")
    n_no = sum(1 for r in rows if r["verdict"] in ("NO-FIT", "UNSUPPORTED", "TIMEOUT"))
    lines.append(f"**Result**: {n_fit} FIT, {n_tight} TIGHT, {n_no} NO-FIT/UNSUPPORTED/TIMEOUT "
                  f"of {len(rows)} cells")
    lines.append("")

    (run_dir / "fit_report.md").write_text("\n".join(lines) + "\n")

    if not args.quiet:
        print(f"\n[validate] {len(rows)} cells → {run_dir / 'fit_report.json'}")
        print(f"[validate] report: {run_dir / 'fit_report.md'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
