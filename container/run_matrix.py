#!/usr/bin/env python3
"""run_matrix.py — orchestrate the GPU inference benchmark matrix.

Runs INSIDE the vLLM container. For each (model, config) cell:
  1. Download model weights to /hf-cache (idempotent; skipped if present).
  2. Start `vllm serve` with the config's flags; wait for /health.
  3. On startup failure, parse the server log → mark cell skipped:<reason>.
  4. On success, run the concurrency sweep (C=1,4,8,16), calling
     `vllm bench serve` per level with a 1 Hz GPU telemetry sample in flight.
  5. Kill the server; weights are KEPT for re-runs (delete with --delete-weights
     or run clean.sh).

Output (in a per-run dir /results/<YYYYMMDD-HHMMSS>_<gpu-slug>/, created
after GPU selection so repeated runs never collide; /results/.latest holds
the run-id basename for entrypoint.sh / bench.sh):
  environment.json                     host + GPU + image + vLLM versions
  server_<model>_<config>.log          raw vLLM server log per cell
  bench_<model>_<config>_<C>.json      raw `vllm bench serve` output
  telemetry_<model>_<config>_<C>.json  1 Hz GPU samples for that level
  cells.json                           manifest (workload/models/cells) consumed by report.py

Usage (from entrypoint.sh):
  python3 /bench/container/run_matrix.py \
      --config /bench/config/models.yaml \
      --results /results \
      --vendor amd [--models M1,M2] [--delete-weights] [--quick]
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
except ImportError:  # pragma: no cover - PyYAML ships in the vllm image
    sys.exit("run_matrix.py: PyYAML is required (pip install pyyaml)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from telemetry import TelemetrySampler, align_window
except ImportError:  # telemetry optional (per-cell GPU metrics → null)
    TelemetrySampler = None
    align_window = None

HF_CACHE = os.environ.get("HF_HOME", "/hf-cache")
HEALTH_PATH = "/health"
DEFAULT_START_TIMEOUT = int(os.environ.get("SERVER_START_TIMEOUT", "900"))

# ── P0-3 compile-detection for per-level warmup ──────────────────────────
# Matches vLLM 0.28.0's JIT-monitor warning (and a few other compilation
# indicators) emitted by the server during the measured window. Only the
# NEW log content appended during a pass is scanned, so startup messages
# like "Kernel JIT monitor activated" are never a false positive.
COMPILE_WARN_RE = re.compile(
    r"JIT compilation during inference",
    re.I)

# ── P0-2 token accounting ───────────────────────────────────────────────
PREFIX_CACHE_RE = re.compile(
    r"enable_prefix_caching=(True|False)")
# ── P0-2 API diagnostic (captured on output-token shortfall) ───────────
DIAG_PROMPT = (
    "Write a concise but complete explanation of how a Transformer model "
    "encodes positional information in the absence of explicit position "
    "embeddings, and compare rotary position embeddings (RoPE) with "
    "learned absolute positions in terms of length generalization. "
    "Use mathematical notation where helpful, but keep the prose "
    "accessible to someone with a graduate-level ML background. "
    "Aim for around four paragraphs. "
    "Do not include any code, and do not use bullet points.")
DIAG_REQUEST_TIMEOUT = 600  # generous: 256 tokens × 3 requests × model latency

# ── T1 AITER attention: numerical-validity capture ────────────────────────
# Three fixed prompts (temperature 0, short output) sent to the server on
# startup for the baseline + aiter-attn cells on AMD. Post-run, token-diff
# the baseline vs aiter outputs: identical = strong agreement; a few late
# flips = float-order tolerance; systematic divergence = stop, don't compare.
AITER_ATTN_CONFIG = "aiter-attn"
AITER_DIAG_PROMPTS = [
    "What is the capital of France? Answer in one short sentence.",
    "List the first five prime numbers, separated by commas. No other text.",
    "Explain, in two sentences, why the sky is blue. No headings.",
]
AITER_DIAG_MAX_TOKENS = 64
AITER_DIAG_REQUESTS = len(AITER_DIAG_PROMPTS)


AITER_EVIDENCE_RE = re.compile(r"aiter", re.I)
TRITON_FALLBACK_RE = re.compile(r"falling back to triton", re.I)
BACKEND_OVERRIDE_RE = re.compile(
    r"Overriding with .* out of potential backends", re.I)


def build_server_env(common: dict, model: dict, cfg: dict) -> dict:
    """Merge the per-layer ``env:`` maps (common < model < cfg) into the env
    overrides for the server process. String values only (env vars are
    strings); non-string values are ignored with a warning. An empty result
    means the server inherits the container env unchanged."""
    out: dict = {}
    for layer in (common.get("env"), model.get("env"), cfg.get("env")):
        for k, v in (layer or {}).items():
            if not isinstance(v, str):
                print(f"[env] WARN ignoring non-string env value {k}={v!r}")
                continue
            out[k] = v
    return out


def scan_attention_evidence(log_path: Path) -> dict:
    """T1: capture attention-backend selection evidence from the server log.

    Returns {aiter_lines: [..<=3], triton_fallback: str|None,
    backend_override: str|None}. Missing log → empty evidence (all None)."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {"aiter_lines": [], "triton_fallback": None,
                "backend_override": None}
    lines = text.splitlines()
    aiter_lines = [l.strip()[:200] for l in lines
                   if AITER_EVIDENCE_RE.search(l)][:3]
    trit = next((l.strip()[:200] for l in lines
                 if TRITON_FALLBACK_RE.search(l)), None)
    override = next((l.strip()[:200] for l in lines
                     if BACKEND_OVERRIDE_RE.search(l)), None)
    return {"aiter_lines": aiter_lines, "triton_fallback": trit,
            "backend_override": override}


def capture_aiter_diag(tag: str, port: int, model_id: str,
                       out_dir: Path) -> Path:
    """T1 numerical-validity capture: send the fixed prompts to the running
    server and save the raw responses. Best-effort: any failure is recorded
    in the file, never fatal."""
    import urllib.parse
    out_path = out_dir / f"diag_aiter_{tag}.json"
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    responses = []
    for i, prompt in enumerate(AITER_DIAG_PROMPTS):
        params = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": AITER_DIAG_MAX_TOKENS,
            "temperature": 0,
        })
        try:
            req = urllib.request.Request(
                url, data=params.encode(),
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(
                    req, timeout=AITER_DIAG_MAX_TOKENS * 5) as r:
                body = json.loads(r.read())
            responses.append({"index": i, "prompt": prompt, "raw": body})
        except Exception as e:
            responses.append({"index": i, "prompt": prompt,
                              "error": str(e)[:400]})
    out_path.write_text(json.dumps(
        {"tag": tag, "model": model_id,
         "num_requests": AITER_DIAG_REQUESTS,
         "max_tokens": AITER_DIAG_MAX_TOKENS, "temperature": 0,
         "purpose": "T1 numerical-validity: token-diff baseline vs aiter-attn",
         "responses": responses}, indent=2))
    return out_path


def effective_prefix_caching(log_path: Path) -> bool | None:
    """Parse the engine config dump for the effective prefix-caching value.

    Returns True/False, or None if the log is missing / unparseable."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    m = PREFIX_CACHE_RE.search(text)
    return m.group(1) == "True" if m else None


def new_log_region(log_path: Path, start_offset: int) -> str:
    """Return bytes written to the log since start_offset."""
    try:
        with open(log_path, "rb") as f:
            f.seek(start_offset)
            data = f.read()
    except OSError:
        return ""
    return data.decode(errors="replace")


def _emit_flag(flag: str, val, extra: list[str], st: dict) -> None:
    """Apply one YAML flag dict entry into extra (and the mutable state st)."""
    if flag == "max-model-len":
        st["max_len"] = "true" if val is True else str(val)
    elif val is True:
        extra.append(f"--{flag}")
    else:
        extra.extend([f"--{flag}", str(val)])


def _apply_flags(flags: dict, extra: list[str], st: dict) -> None:
    """Emit a flags: dict into the extra argv list (and st[max_len]).

    Precedence is handled by the caller: common first, then model, then
    config — each call mutates the same `extra` / `st` so later calls win.
    """
    for flag, val in (flags or {}).items():
        _emit_flag(flag, val, extra, st)


def _build_bench_cmd(model: dict, workload: dict, concurrency: int,
                     out_file: str,
                     num_prompts: int | None = None,
                     num_warmups: int | None = None) -> list[str]:
    """Assemble the ``vllm bench serve`` argv for one concurrency level.

    Optional overrides: ``num_prompts`` and ``num_warmups`` let the caller
    adjust the client-side load (e.g. fewer prompts for a warmup pass)."""
    return [
        "vllm", "bench", "serve",
        "--host", "127.0.0.1", "--port", str(workload.get("port", 8000)),
        "--backend", workload.get("backend", "openai-chat"),
        "--endpoint", workload.get("endpoint", "/v1/chat/completions"),
        "--model", model["id"],
        "--dataset-name", "random",
        "--random-input-len",
        str(workload.get("random_input_len", 512)),
        "--random-output-len",
        str(workload.get("random_output_len", 256)),
        "--num-prompts",
        str(num_prompts or workload.get("num_prompts", 50)),
        "--max-concurrency", str(concurrency),
        "--num-warmups",
        str(num_warmups if num_warmups is not None
            else workload.get("num_warmups", 2)),
        "--seed", str(workload.get("seed", 42)),
        "--temperature", str(workload.get("temperature", 0)),
        "--ignore-eos",
        "--percentile-metrics",
        workload.get("percentile_metrics", "ttft,tpot,itl"),
        "--metric-percentiles",
        workload.get("metric_percentiles", "50,90,99"),
        "--save-result",
        "--result-filename", out_file,
    ]


def _check_tokens(raw: dict, workload: dict,
                  threshold: float = 0.9) -> dict | None:
    """Output-token accounting guard. Returns check dict or None."""
    try:
        expected = (
            int(workload.get("num_prompts", 50))
            * int(workload.get("random_output_len", 256)))
    except (TypeError, ValueError):
        return None
    actual = raw.get("total_output_tokens")
    if actual is None or expected <= 0:
        return None
    ratio = actual / expected
    check: dict = {"expected": expected, "actual": actual, "ratio": round(ratio, 3)}
    if ratio < threshold:
        check["flag"] = "output-token-shortfall"
    return check


def _diagnose_shortfall(
    tag: str, c: int, port: int, model_id: str,
    workload: dict, out_dir: Path) -> Path:
    """Send N chat completions and save raw API responses.

    Captures finish_reason, usage.prompt_tokens, usage.completion_tokens,
    usage.completion_tokens_details (reasoning_tokens) when present.
    Best-effort: any failure is logged and the file still gets written.
    """
    import urllib.parse
    out_path = out_dir / f"diag_{tag}_{c}.json"
    n = 3
    params = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": DIAG_PROMPT}],
        "max_tokens": int(workload.get("random_output_len", 256)),
        "temperature": workload.get("temperature", 0),
        "ignore_eos": True,
    })
    responses = []
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    for i in range(n):
        try:
            req = urllib.request.Request(
                url, data=params.encode(),
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=DIAG_REQUEST_TIMEOUT) as r:
                body = json.loads(r.read())
                responses.append({"index": i, "raw": body})
        except Exception as e:
            responses.append({"index": i, "error": str(e)[:400]})
    out_path.write_text(json.dumps({"tag": tag, "concurrency": c,
                                    "num_requests": n,
                                    "responses": responses}, indent=2))
    return out_path


def build_server_cmd(model: dict, cfg: dict, common: dict) -> list[str]:
    """Assemble the ``vllm serve ...`` argv for one (model, config) cell.

    Precedence: common < model < config (each later ``flags:`` dict overrides).
    ``flags:`` entries in any layer become CLI args; ``true`` → bare flag.

    P0 note (2026-09-08 review): the common layer now carries
    ``no-enable-prefix-caching: true`` — vLLM 0.28.0 defaults to
    enable_prefix_caching=True and the old config only *commented* that it
    was off, so prefix caching was actually ON in the 2026-09-03 run (up to
    76% hit rate on M2/AMD at C=16). run_matrix.py verifies the effective
    value in the server log and fails the cell if it comes back True.

    ``max_num_seqs:`` (model or common) — emitted as ``--max-num-seqs``. The
    workload tops out at C=16, and hybrid Mamba/GDN models hard-fail startup
    when max_num_seqs exceeds their Mamba cache block count (e.g. M4 on a
    32 GB card: 21 blocks), so the matrix pins it to the workload ceiling.
    ``max-model-len`` — model defaults to config; long-context can override.
    """
    st: dict = {"max_len": str(
        model.get("max_model_len", common.get("max-model-len", 8192)))}
    extra: list[str] = []
    # common flags first (lowest priority)
    _apply_flags(common.get("flags"), extra, st)
    # model flags
    _apply_flags(model.get("flags"), extra, st)
    # config flags (highest priority — override model + common)
    _apply_flags(cfg.get("flags"), extra, st)
    max_len = st["max_len"]

    cmd = ["vllm", "serve", model["id"],
           "--host", str(common.get("host", "0.0.0.0")),
           "--port", str(common.get("port", 8000)),
           "--max-model-len", max_len,
           "--gpu-memory-utilization",
           str(model.get("gpu_memory_utilization",
                         common.get("gpu_memory_utilization", 0.90)))]
    max_num_seqs = model.get("max_num_seqs", common.get("max_num_seqs"))
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    if common.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    return cmd + extra


def build_bench_cmd(model: dict, workload: dict, concurrency: int,
                    out_file: str) -> list[str]:
    """Assemble the ``vllm bench serve ...`` argv for one concurrency level."""
    return _build_bench_cmd(model, workload, concurrency, out_file)


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


# ── Intel XPU enumeration (torch.xpu — the runtime vLLM itself uses) ─────────
_XPU_PROBE = (
    "import json, torch\n"
    "try:\n"
    "    n = torch.xpu.device_count() if torch.xpu.is_available() else 0\n"
    "except Exception:\n"
    "    n = 0\n"
    "print(json.dumps([{'index': i,\n"
    "                   'name': torch.xpu.get_device_name(i),\n"
    "                   'total_bytes': torch.xpu.mem_get_info(i)[1]}\n"
    "                  for i in range(n)]))"
)
_XPU_DEVS: list[dict] | None = None

# Probe the *filtered* view under the current ONEAPI_DEVICE_SELECTOR: a
# malformed selector makes the SYCL runtime abort the process at init
# ("Incomplete selector!" / "Backend is required but missing"), so the
# subprocess dies non-zero and prints nothing usable.
_XPU_SEL_PROBE = (
    "import json, torch\n"
    "try:\n"
    "    n = torch.xpu.device_count() if torch.xpu.is_available() else 0\n"
    "    name = torch.xpu.get_device_name(0) if n else ''\n"
    "    print(json.dumps({'n': n, 'name': name}))\n"
    "except Exception:\n"
    "    print(json.dumps({'n': -1, 'name': ''}))"
)


def _xpu_probe_selection(timeout: float = 180.0) -> tuple[int, str] | None:
    """(device_count, name_of_device_0) under the current env (selector
    applied), or None if the SYCL runtime rejected the process (bad
    selector format aborts before any output)."""
    out = _run_cmd(["python3", "-c", _XPU_SEL_PROBE], timeout=timeout)
    if not out:
        return None
    try:
        d = json.loads(out.strip().splitlines()[-1])
        return int(d["n"]), str(d.get("name", ""))
    except (ValueError, KeyError, IndexError):
        return None


def _xpu_devices(timeout: float = 180.0) -> list[dict]:
    """Enumerate XPU devices via torch.xpu (PyTorch XPU / Level Zero).

    The vLLM XPU image ships a SYCL PyTorch build, so torch.xpu is the
    exact runtime `vllm serve` uses — no dependency on optional CLIs
    (xpu-smi, zeinfo) that the image may lack.

    Returns [{'index': <Level Zero index>, 'name': str,
              'total_bytes': int}, ...]. Empty list if torch.xpu is
    unavailable or no device is visible (e.g. the DRI render node was
    not passed into the container).

    Probed once per process, before select_gpu() applies
    ONEAPI_DEVICE_SELECTOR, so indices are *physical* (unfiltered) — the
    same contract as the AMD path (unfiltered rocm-smi table).
    """
    global _XPU_DEVS
    if _XPU_DEVS is None:
        out = _run_cmd(["python3", "-c", _XPU_PROBE], timeout=timeout)
        try:
            devs = json.loads(out or "[]")
            _XPU_DEVS = devs if isinstance(devs, list) else []
        except json.JSONDecodeError:
            _XPU_DEVS = []
    return _XPU_DEVS


def slugify(name: str) -> str:
    """Lowercase, non-alphanumeric → hyphen."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "gpu"


def gpu_name(vendor: str, idx: int | None = None) -> str:
    """Best-effort GPU model name for the per-run directory slug.

    Never raises — falls back to ``<vendor>-gpu`` if no CLI is available
    or the output is unparseable.
    """
    vendor = (vendor or "auto").lower()
    if vendor == "nvidia":
        i = idx if idx is not None else 0
        out = _run_cmd(["nvidia-smi", "--query-gpu=name",
                        "--format=csv,noheader", "-i", str(i)])
        return (out or "").strip() or f"nvidia-gpu"
    if vendor == "amd":
        env = {k: v for k, v in os.environ.items()
               if k != "HIP_VISIBLE_DEVICES"}
        out = _run_cmd(["rocm-smi", "--showproductname"], env=env,
                       timeout=30.0)
        if out:
            pat = rf"GPU\[{idx}\]" if idx is not None else r"GPU\[\d+\]"
            for line in out.splitlines():
                m = re.match(rf"{pat}\s*:\s*(.+)", line)
                if m:
                    name = m.group(1).strip()
                    # first line after the vendor name; if it looks like an
                    # error (e.g. "get_name, Error when calling libdrm")
                    # or N/A, keep the first non-error field as fallback
                    if "Error" not in name and "N/A" not in name:
                        # strip "Card Model:" prefix + padding (some rocm-smi
                        # versions emit the raw field name when the product DB
                        # lacks this device ID, e.g. "Card Model:  0x7551")
                        name = re.sub(
                            r"^\s*(?:Card\s+Model\s*:\s*)", "", name).strip()
                        return name or f"amd-gpu"
        return f"amd-gpu"
    if vendor == "intel":
        devs = _xpu_devices()
        d = None
        if devs:
            if idx is not None and 0 <= idx < len(devs):
                d = devs[idx]
            else:
                d = max(devs, key=lambda x: x.get("total_bytes") or 0)
        if d and d.get("name"):
            return d["name"]
        # fallback: xpu-smi (not always in the image)
        out = _run_cmd(["xpu-smi", "discovery"], timeout=30.0)
        if out:
            m = re.search(
                r"^\s*\|(?:\s*(?:iGPU|dGPU|Discrete GPU))"
                r"\s*\|\s*\d+\s*\|\s*([^|]+?)\s*\|",
                out, re.M)
            if m:
                return m.group(1).strip()
        return "intel-gpu"
    return f"{vendor}-gpu"


# ── Environment metadata (→ environment.json) ─────────────────────────────────
def _first_line(cmd: list[str]) -> str | None:
    out = _run_cmd(cmd)
    if not out:
        return None
    for line in out.splitlines():
        if line.strip():
            return line.strip()
    return None


def _read_file(path: str) -> str | None:
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _torch_stack() -> tuple[str | None, str | None]:
    """(cuda, hip) runtime versions from the image's torch, best-effort."""
    out = _run_cmd(["python3", "-c",
                    "import torch; print(repr(torch.version.cuda or '')); "
                    "print(repr(torch.version.hip or ''))"], timeout=60.0)
    lines = (out or "").splitlines()
    cuda = lines[0].strip("'\" ") if len(lines) > 0 else None
    hip = lines[1].strip("'\" ") if len(lines) > 1 else None
    return (cuda or None, hip or None)


def collect_environment(vendor: str, gpu_index: int | None, gpu: str) -> dict:
    """Best-effort environment metadata for environment.json. Never raises.

    Host-side facts that the container cannot see (host OS, docker version,
    image name/digest) come via env vars set by bench.sh.
    """
    vendor = (vendor or "auto").lower()
    env: dict = {
        "vendor": vendor,
        "gpu": gpu,
        "gpu_index_in_container": gpu_index,
        "vram_total_gb": None,
        "driver": None,
        "stack": {"cuda": None, "rocm": None, "xpu": None},
        "os": os.environ.get("HOST_OS") or None,
        "kernel": _first_line(["uname", "-r"]),
        "cpu": None,
        "cpu_cores": os.cpu_count(),
        "ram_gb": None,
        "gpu_kernel_modules": [],
        "docker_version": os.environ.get("DOCKER_VERSION") or None,
        "image": os.environ.get("IMAGE") or None,
        "image_id": os.environ.get("IMAGE_DIGEST") or None,
        "vllm_version": None,
    }
    if vendor == "nvidia":
        i = str(gpu_index if gpu_index is not None else 0)
        out = _run_cmd(["nvidia-smi", "--query-gpu=driver_version,memory.total",
                        "--format=csv,noheader,nounits", "-i", i], timeout=30.0)
        if out:
            parts = [p.strip() for p in out.split(",")]
            if parts and parts[0]:
                env["driver"] = parts[0]
            if len(parts) > 1:
                try:
                    env["vram_total_gb"] = round(float(parts[1]) / 1024.0, 1)
                except ValueError:
                    pass
    elif vendor == "amd":
        env2 = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
        out = _run_cmd(["rocm-smi", "--showdriverversion"], env=env2, timeout=30.0)
        m = re.search(r"(\d+\.\d+(?:\.\d+)?)", out or "")
        if m:
            env["driver"] = m.group(1)
        out = _run_cmd(["rocm-smi", "--showmeminfo", "vram"], env=env2, timeout=30.0)
        pat = (rf"GPU\[{gpu_index}\].*?VRAM Total Memory \(B\): (\d+)"
               if gpu_index is not None
               else r"GPU\[\d+\].*?VRAM Total Memory \(B\): (\d+)")
        m = re.search(pat, out or "")
        if m:
            env["vram_total_gb"] = round(int(m.group(1)) / 1024 ** 3, 1)
        # T1: AITER version (best-effort; import can be slow on AMD stack)
        av = _run_cmd(
            ["python3", "-c",
             "import aiter; print(getattr(aiter, '__version__', 'unknown'))"],
            timeout=120.0)
        if av:
            env["stack"]["aiter"] = av.strip().splitlines()[-1]
    elif vendor == "intel":
        devs = _xpu_devices()
        d = None
        if devs:
            if gpu_index is not None and 0 <= gpu_index < len(devs):
                d = devs[gpu_index]
            else:
                d = devs[0]
        if d and d.get("total_bytes"):
            env["vram_total_gb"] = round(d["total_bytes"] / 1024 ** 3, 1)
        # driver: xe for Arc A/B dGPUs (B70 et al.), i915 for legacy;
        # xpu-smi discovery as last resort.
        for mod in ("xe", "i915"):
            v = (_read_file(f"/sys/module/{mod}/version") or "").strip()
            if v:
                env["driver"] = f"{mod} {v}"
                env["stack"]["xpu"] = f"{mod} {v}"
                break
        if not env["driver"]:
            out = _run_cmd(["xpu-smi", "discovery"], timeout=30.0)
            m = re.search(
                r"^\s*\|(?:\s*(?:iGPU|dGPU|Discrete GPU))\s*\|\s*\d+\s*\|[^|]*\|\s*([^|]+?)\s*\|",
                out or "", re.M)
            if m:
                env["driver"] = m.group(1)
                env["stack"]["xpu"] = m.group(1)
        # Telemetry source probe (best-effort, never fatal): record which
        # sources this host's GPU telemetry will come from — xe sysfs
        # (freq/temp/power) and/or xpu-smi (util/mem) — so a run that
        # reports "n/a" power says so explicitly instead of silently.
        if TelemetrySampler is not None:
            try:
                probe_sampler = TelemetrySampler("intel", gpu_index)
                if probe_sampler.probe() is not None:
                    src = probe_sampler.metric_sources()
                    env["telemetry_metrics"] = {
                        m: src.get(m, "none")
                        for m in ("frequency", "temperature", "power",
                                  "utilization", "memory")}
            except Exception:
                pass

    cuda, hip = _torch_stack()
    env["stack"]["cuda"] = cuda
    env["stack"]["rocm"] = hip

    m = re.search(r"model name\s*:\s*(.+)", _read_file("/proc/cpuinfo") or "")
    if m:
        env["cpu"] = m.group(1).strip()
    m = re.search(r"MemTotal:\s*(\d+)", _read_file("/proc/meminfo") or "")
    if m:
        env["ram_gb"] = round(int(m.group(1)) / 1024 / 1024, 1)
    mod_names = {l.split()[0] for l in (_read_file("/proc/modules") or "")
                 .splitlines() if l.split()}
    known = ("nvidia", "nvidia_uvm", "nvidia_drm", "nvidia_modeset",
             "amdgpu", "amdkcl", "xe", "i915")
    env["gpu_kernel_modules"] = sorted(mod_names & set(known))

    out = _run_cmd(["vllm", "--version"], timeout=60.0)
    m = re.search(r"(\d+\.\d+\.\d+\S*)", out or "")
    if m:
        env["vllm_version"] = m.group(1)
    return env


def select_gpu(vendor: str, forced_idx: int | None = None) -> int:
    """Select a GPU inside the container and set the visibility env var.

    Returns the index to pass to the telemetry sampler:
      * AMD: the physical index (vllm sees it as its device 0 after
        HIP_VISIBLE_DEVICES; the sampler reads the *unfiltered* rocm-smi
        table, so it needs the physical index to pick the right row).
      * NVIDIA: 0 (bench.sh passes --gpus device=N, so only one GPU is
        visible in the container and it is device 0).
      * Intel: the physical Level Zero index of the selected device
        (auto-picked by VRAM so a dGPU wins over an iGPU); applied
        via ONEAPI_DEVICE_SELECTOR=level_zero/<idx> (colon syntax — the
        current oneAPI/SYCL runtime rejects the older slash-based forms;
        a verification probe confirms the runtime accepts it before any
        server is launched).
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
        devs = _xpu_devices()
        if not devs:
            raise SystemExit(
                "ERROR: no XPU device visible in the container "
                "(torch.xpu.device_count() == 0). Check on the host:\n"
                "  1. xe kernel module loaded:  lsmod | grep xe\n"
                "  2. DRI nodes present:        ls -l /dev/dri\n"
                "  3. bench.sh --dry-run shows: -v /dev/dri:/dev/dri\n"
                "  4. inside this container:    zeinfo (should list the card)")
        if forced_idx is not None and not 0 <= forced_idx < len(devs):
            raise SystemExit(f"ERROR: XPU index {forced_idx} not present "
                             f"(available: {list(range(len(devs)))})")
        idx = (forced_idx if forced_idx is not None
               else max(range(len(devs)),
                        key=lambda i: devs[i].get("total_bytes") or 0))
        d = devs[idx]
        # Colon syntax: 'level_zero:<idx>'. The OneAPI/SYCL runtime rejects
        # the older 'level_zero/<idx>' and 'level_zero/<idx>:*' forms
        # ("Incomplete selector!" / "Backend is required but missing").
        os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{idx}"
        # Verify the SYCL runtime accepts the selector before launching any
        # server. A malformed value aborts the process at device init, which
        # would otherwise surface as one crashed vLLM server per cell instead
        # of one failed probe here (~15s).
        sel = _xpu_probe_selection()
        if sel is None or sel[0] != 1:
            raise SystemExit(
                f"ERROR: ONEAPI_DEVICE_SELECTOR=level_zero:{idx} not accepted "
                f"by the SYCL runtime (probe saw {sel}). The selector format "
                f"varies across oneAPI releases — try 'level_zero/{idx}:*' or "
                f"SYCL_DEVICE_FILTER inside the container.")
        total_gb = (d.get("total_bytes") or 0) // (1024 ** 3)
        print(f"[gpu-select] Intel idx {idx} ({d.get('name') or 'XPU'}, "
              f"{total_gb} GB) → ONEAPI_DEVICE_SELECTOR=level_zero:{idx} "
              f"(verified: {sel[0]} device visible, {sel[1]})")
        return idx
    raise SystemExit(f"ERROR: unsupported GPU_VENDOR {vendor!r} "
                     f"(expect amd|nvidia|intel)")


def _expand_model_keys(raw: str, known_keys: list[str]) -> list[str]:
    """Expand a comma-/range-separated model list into concrete keys.

    Supports: ``M1,M2``, ``M1-M4``, ``M2,M3-M4``, ``M4-M2`` (reverse order
    is allowed — it just follows the config's natural order).
    Unknown keys are passed through unchanged.
    """
    result: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.match(r"^(M\d+)-(M\d+)$", token)
        if m:
            start_key, end_key = m.group(1), m.group(2)
            start_idx = next(i for i, k in enumerate(known_keys) if k == start_key)
            end_idx = next(i for i, k in enumerate(known_keys) if k == end_key)
            result.extend(known_keys[start_idx:end_idx + 1] if start_idx <= end_idx
                          else known_keys[end_idx:start_idx + 1])
        else:
            result.append(token)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU inference benchmark orchestrator (runs inside the vLLM "
                    "container). For each (model, config) cell: start the server, "
                    "wait for /health, run the concurrency sweep with 1 Hz GPU "
                    "telemetry, then kill the server. Weights are kept in the HF "
                    "cache by default so re-runs skip the re-download.")
    p.add_argument("--config", default="/bench/config/models.yaml",
                   help="path to models.yaml")
    p.add_argument("--results", default="/results",
                   help="output root dir (a per-run subdir is created inside)")
    p.add_argument("--vendor", default=os.environ.get("GPU_VENDOR", "auto"),
                   choices=["auto", "nvidia", "amd", "intel"],
                   help="GPU vendor (auto-detects from telemetry CLIs)")
    p.add_argument("--gpu-index", type=int, default=None,
                   help="physical GPU index (overrides auto-pick by VRAM)")
    p.add_argument("--models", default=None,
                   help="comma-/range-separated model list, e.g. M1,M2 or M1-M4")
    p.add_argument("--configs", default=None, help="comma list, e.g. baseline,kv-fp8")
    p.add_argument("--concurrency", default=None, help="comma list, e.g. 1,8,16")
    p.add_argument("--keep-weights", action="store_true",
                   help="no-op, kept for backward compatibility "
                        "(weights are kept by default now)")
    p.add_argument("--delete-weights", action="store_true",
                   help="delete model weights from the HF cache after each "
                        "model (old behavior; off by default so re-runs skip "
                        "the ~20-25 GB re-download). Use clean.sh for a one-shot "
                        "manual cleanup instead.")
    p.add_argument("--quick", action="store_true",
                   help="M1 only, baseline+kv-fp8, concurrency 1,8")
    p.add_argument("--start-timeout", type=int, default=DEFAULT_START_TIMEOUT,
                   help="server health-wait budget in seconds (default 900)")
    p.add_argument("--dry-run", action="store_true",
                   help="print planned server/bench commands, run nothing")
    return p.parse_args()


# ── Server lifecycle ─────────────────────────────────────────────────────────
def wait_health(port: int, timeout: float,
                proc: subprocess.Popen | None = None) -> bool:
    url = f"http://127.0.0.1:{port}{HEALTH_PATH}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # server already exited — don't poll a dead port
        try:
            with urlrequest.urlopen(url, timeout=5) as r:
                if 200 <= r.status < 300:
                    return True
        except (URLError, OSError, ValueError):
            pass
        time.sleep(5)
    return False


def start_server(cmd: list[str], log_path: Path,
                 env: dict | None = None) -> subprocess.Popen:
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
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

# oneCCL log lines (e.g. "|CCL_WARN| pidfd is not supported, fallbacks to
# drmfd exchange mode") are benign IPC-fallback noise.  The "not supported"
# substring there would falsely match the unsupported patterns.
def _strip_ccl_lines(text: str) -> str:
    return re.sub(r"[^\n]*\|CCL_(?:WARN|INFO|ERROR)\|[^|\n]*\n?", "", text)


def parse_skip_reason(log_path: Path) -> tuple[str, str]:
    """Return (status, reason) from a failed server log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return "failed", "no-server-log"
    # CCL lines can contain "not supported" (benign pidfd->drmfd fallback);
    # scan the CCL-stripped text for model/quant failure patterns.
    clean = _strip_ccl_lines(text)
    if OOM_PATTERNS.search(clean):
        return "skipped", "oom"
    if re.search(r"kv[ -]?cache.{0,40}(fp8|dtype)|fp8.{0,40}(not support|unavail)",
                 clean, re.I):
        return "skipped", "kv-fp8-unsupported"
    m = UNSUPPORTED_PATTERNS.search(clean)
    if m:
        line = clean[max(0, m.start() - 60):m.end() + 40].replace("\n", " ")
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
def _build_telemetry(sampler, samples, wall_s, duration_s,
                     output_tokens, token_flagged):
    """P1: telemetry with power/energy aligned to the measured bench window.

    Top-level aggregates (power_avg_w, util, mem, energy_j) are computed
    over the aligned window; ``full_window`` keeps the pre-alignment values
    for traceability; raw 1 Hz samples are archived so future re-aggregation
    is possible. Returns None when the sampler is unavailable or produced
    no samples.
    """
    if sampler is None or not samples:
        return None
    window = {
        "aligned": False, "reason": None,
        "wall_s": round(wall_s, 2),
        "measured_window_s": (round(duration_s, 2)
                              if duration_s is not None else None),
        "n_samples": len(samples), "n_samples_full": len(samples),
    }
    aligned = samples
    if duration_s is None:
        window["reason"] = "no-duration"
    elif align_window is None:
        window["reason"] = "align-unavailable"
    else:
        aligned = align_window(samples, wall_s, duration_s)
        if aligned is None:
            window["reason"] = "insufficient-samples"
        else:
            window["aligned"] = True
            window["n_samples"] = len(aligned)
    base = sampler.aggregate(aligned)
    if base is None:
        return None
    full = sampler.aggregate(samples) or {}
    telem = dict(base)
    telem["window"] = window
    # GPU energy per 1000 output tokens (null on token shortfall flags: a
    # short run's J/ktok is meaningless; the flag marks the level).
    e = telem.get("energy_j")
    telem["energy_j_per_ktok"] = (
        round(e / (output_tokens / 1000.0), 1)
        if (e is not None and output_tokens and not token_flagged) else None)
    telem["full_window"] = {
        "power_avg_w": full.get("power_avg_w"),
        "power_peak_w": full.get("power_peak_w"),
        "energy_j": full.get("energy_j"),
    }
    telem["samples"] = samples
    return telem


def run_cell(model_key: str, model: dict, cfg: dict, workload: dict, common: dict,
             results: Path, vendor: str, gpu_index: int | None, concurrencies: list[int],
             start_timeout: int, dry_run: bool) -> dict:
    tag = f"{model_key}_{cfg['name']}"
    port = int(common.get("port", 8000))
    s_cmd = build_server_cmd(model, cfg, common)
    # T1: env: layers (common < model < cfg) apply to the server process only
    env_overrides = build_server_env(common, model, cfg)
    # T1: aiter-attn is AMD-only — explicit skip on other vendors
    if cfg["name"] == AITER_ATTN_CONFIG and vendor in ("nvidia", "intel"):
        print(f"[cell {tag}] skipped: aiter-attn is AMD-only (vendor={vendor})")
        return {"model": model_key, "model_id": model["id"],
                "config": cfg["name"], "status": "skipped",
                "reason": "aiter-amd-only", "server_flags": {},
                "server_env": {}, "server_log": None,
                "concurrency_results": {}}
    print(f"[cell {tag}] server: {' '.join(s_cmd)}")
    if env_overrides:
        print(f"[cell {tag}] server env: "
              f"{' '.join(f'{k}={v}' for k, v in env_overrides.items())}")

    cell: dict = {
        "model": model_key, "model_id": model["id"], "config": cfg["name"],
        "status": "failed", "reason": None,
        "server_flags": {
            "max_model_len": model.get("max_model_len", common.get("max-model-len", 8192)),
            "gpu_memory_utilization": model.get("gpu_memory_utilization",
                                                common.get("gpu_memory_utilization", 0.90)),
            "max_num_seqs": model.get("max_num_seqs",
                                      common.get("max_num_seqs", 256)),
            **{f.replace("-", "_"): v for f, v in (cfg.get("flags") or {}).items()},
        },
        "server_log": None,
        "server_env": env_overrides,
        "concurrency_results": {},
    }
    # ── P0-3 protocol: measured passes (with controlled server restarts)
    #    and per-level warmup until no new kernel JIT compilations appear.
    num_passes = max(1, int(workload.get("num_passes", 1)))
    warmup_max = max(0, int(workload.get("warmup_max_passes", 0)))
    warmup_prompts = int(workload.get("warmup_prompts", 8))
    token_threshold = float(workload.get("token_check_threshold", 0.9))
    requested_off_prefix = "--no-enable-prefix-caching" in s_cmd

    if dry_run:
        for c in concurrencies:
            w_cmd = _build_bench_cmd(model, workload, c,
                                     f"warmup_{tag}_{c}_1.json",
                                     num_prompts=warmup_prompts,
                                     num_warmups=0)
            b_cmd = build_bench_cmd(model, workload, c, f"bench_{tag}_{c}_pass1.json")
            print(f"[cell {tag}]   warmup C={c}: {' '.join(w_cmd)}")
            print(f"[cell {tag}]   bench  C={c} (pass 1/{num_passes}): "
                  f"{' '.join(b_cmd)}")
        cell["status"] = "dry-run"
        cell["protocol"] = {
            "num_passes": num_passes,
            "warmup_max_passes": warmup_max,
            "prefix_caching": ("disabled (--no-enable-prefix-caching)"
                               if requested_off_prefix else "NOT explicitly off"),
        }
        return cell

    cell["protocol"] = {
        "num_passes": num_passes,
        "warmup_max_passes": warmup_max,
        "prefix_caching": ("disabled (--no-enable-prefix-caching)"
                           if requested_off_prefix else "NOT explicitly off"),
    }
    cell["server_logs"] = {}
    diag_done = False
    sampler = (TelemetrySampler(vendor, gpu_index) if TelemetrySampler else None)
    server_env = None
    if env_overrides:
        server_env = dict(os.environ)
        server_env.update(env_overrides)

    def _sweep_pass(pass_i: int) -> str | None:
        """Run one full pass (all concurrency levels). Returns an error
        string to abort the cell, or None on success."""
        nonlocal diag_done
        server_log_i = results / f"server_{tag}_pass{pass_i}.log"
        cell["server_logs"][str(pass_i)] = server_log_i.name
        if pass_i == 1:
            cell["server_log"] = server_log_i.name  # legacy key
        server_p = start_server(s_cmd, server_log_i, env=server_env)
        try:
            if not wait_health(port, start_timeout, proc=server_p):
                st, reason = parse_skip_reason(server_log_i)
                print(f"[cell {tag}] pass {pass_i}: server FAILED → "
                      f"{st}:{reason}")
                return f"pass{pass_i}-{st}:{reason}"
            # P0-1: verify the effective prefix-caching value in the log.
            eff_pc = effective_prefix_caching(server_log_i)
            if pass_i == 1:
                cell["server_flags"]["effective_prefix_caching"] = eff_pc
            if requested_off_prefix and eff_pc is True:
                reason = ("prefix-caching-not-disabled (server log shows "
                          "enable_prefix_caching=True despite "
                          "--no-enable-prefix-caching)")
                print(f"[cell {tag}] pass {pass_i}: {reason}")
                return reason
            if eff_pc is None:
                print(f"[cell {tag}] pass {pass_i}: WARN could not parse "
                      f"effective enable_prefix_caching from server log")
            elif eff_pc is True:
                print(f"[cell {tag}] pass {pass_i}: WARN prefix caching is "
                      f"EFFECTIVELY ON — throughput may be inflated by "
                      f"prompt replay across concurrency levels")
            # T1: capture attention-backend selection evidence (pass 1)
            if pass_i == 1:
                cell["server_flags"]["attention_evidence"] = (
                    scan_attention_evidence(server_log_i))
                ev = cell["server_flags"]["attention_evidence"]
                if cfg["name"] == AITER_ATTN_CONFIG:
                    note = ("; ".join(ev["aiter_lines"]) or
                            "NO AITER EVIDENCE IN LOG — likely silent "
                            "fallback to baseline backend")
                    print(f"[cell {tag}] attention evidence: {note}")
                if (vendor == "amd"
                        and cfg["name"] in ("baseline", AITER_ATTN_CONFIG)):
                    cell["aiter_diag"] = capture_aiter_diag(
                        tag, port, model["id"], results).name
                    print(f"[cell {tag}] numerical-validity capture → "
                          f"{cell['aiter_diag']}")
            print(f"[cell {tag}] pass {pass_i}/{num_passes}: server healthy "
                  f"→ sweeping C={concurrencies}")
            for c in concurrencies:
                level = cell["concurrency_results"].setdefault(str(c), {
                    "status": "ok", "reason": None, "passes": [],
                })
                # ── P0-3 warmup: replay the same load shape until no NEW
                # kernel JIT compilations appear in the server log (or
                # warmup_max passes are exhausted).
                warmup_rec = {"max_passes": warmup_max, "passes": 0,
                              "stable": False, "compilations": []}
                for n in range(1, warmup_max + 1):
                    off = (server_log_i.stat().st_size
                           if server_log_i.exists() else 0)
                    w_out = results / f"warmup_{tag}_{c}_{n}.json"
                    w_cmd = _build_bench_cmd(model, workload, c,
                                             str(w_out),
                                             num_prompts=warmup_prompts,
                                             num_warmups=0)
                    print(f"[cell {tag}]   warmup C={c} pass {n} "
                          f"({warmup_prompts} prompts) ...")
                    subprocess.run(w_cmd, capture_output=True, text=True)
                    new = new_log_region(server_log_i, off)
                    comps = COMPILE_WARN_RE.findall(new)
                    warmup_rec["passes"] = n
                    warmup_rec["compilations"].extend(comps)
                    if not comps:
                        warmup_rec["stable"] = True
                        break
                    print(f"[cell {tag}]   warmup C={c}: {len(comps)} new "
                          f"JIT compilation(s) — continuing")
                level["warmup"] = warmup_rec
                # ── P0-3 measured bench (telemetry around the run).
                off = (server_log_i.stat().st_size
                       if server_log_i.exists() else 0)
                bench_out = results / f"bench_{tag}_{c}_pass{pass_i}.json"
                b_cmd = build_bench_cmd(model, workload, c, str(bench_out))
                print(f"[cell {tag}]   bench C={c} pass {pass_i} ...")
                samples = []
                if sampler:
                    samples = sampler.start()
                t0 = time.time()
                proc = subprocess.run(b_cmd, capture_output=True, text=True)
                wall = time.time() - t0
                if sampler:
                    samples = sampler.stop()
                # P1: parse the bench JSON first — its `duration` is the
                # measured window the power samples must be aligned to.
                bench_raw: dict = {}
                tc: dict | None = None
                if proc.returncode == 0 and bench_out.exists():
                    try:
                        bench_raw = json.loads(bench_out.read_text())
                        tc = _check_tokens(bench_raw, workload,
                                           token_threshold)
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"[cell {tag}]   WARN C={c} pass {pass_i}: "
                              f"bench JSON error: {e}")
                telem_out = results / \
                    f"telemetry_{tag}_{c}_pass{pass_i}.json"
                telem = _build_telemetry(
                    sampler, samples, wall,
                    bench_raw.get("duration") if bench_raw else None,
                    bench_raw.get("total_output_tokens") if bench_raw else None,
                    bool(tc and tc.get("flag")))
                if telem is not None:
                    telem_out.write_text(json.dumps(telem, indent=2))
                pass_entry: dict = {
                    "pass": pass_i,
                    "bench_json": bench_out.name,
                    "telemetry_json": (telem_out.name if telem else None),
                    "wall_time_s": round(wall, 1),
                    "status": "ok", "reason": None,
                    "jit_during_bench": bool(
                        COMPILE_WARN_RE.search(new_log_region(server_log_i, off))),
                    "token_check": tc,
                }
                if pass_entry["jit_during_bench"]:
                    print(f"[cell {tag}]   WARN C={c} pass {pass_i}: kernel "
                          f"JIT compilation inside the measured window")
                if proc.returncode != 0 or not bench_out.exists():
                    pass_entry["status"] = "failed"
                    pass_entry["reason"] = (proc.stderr or proc.stdout or "")[-400:]
                elif tc and tc.get("flag") and not diag_done:
                    # P0-2: flag only — the numbers are NEVER "corrected".
                    print(f"[cell {tag}]   output-token shortfall at C={c} "
                          f"({tc['actual']}/{tc['expected']}) → capturing "
                          f"API diagnostic")
                    _diagnose_shortfall(tag, c, port, model["id"],
                                        workload, results)
                    diag_done = True
                level["passes"].append(pass_entry)
            return None
        finally:
            stop_server(server_p)

    try:
        for pass_i in range(1, num_passes + 1):
            err = _sweep_pass(pass_i)
            if err is not None:
                cell["status"] = "failed"
                cell["reason"] = err
                # keep whatever earlier passes produced (report.py handles
                # partial pass sets); stop the remaining passes.
                break
        # per-level status from the pass entries
        for c_str, level in cell["concurrency_results"].items():
            passes = level.get("passes", [])
            ok = [p for p in passes if p["status"] == "ok"]
            if not passes:
                level["status"], level["reason"] = "failed", "no-passes-run"
            elif not ok:
                level["status"] = "failed"
                level["reason"] = passes[-1].get("reason")
            elif len(ok) < len(passes):
                level["status"] = "degraded"
                level["reason"] = (f"{len(ok)}/{len(passes)} passes ok; "
                                   f"last failure: "
                                   f"{(passes[-1].get('reason') or '')[:120]}")
            else:
                level["status"], level["reason"] = "ok", None
    except Exception as e:  # defensive: never lose a cell to an exception
        cell["status"], cell["reason"] = "failed", f"orchestration:{e}"
    return cell


def run_model(model_key: str, model: dict, cfgs: list[dict], workload: dict,
              common: dict, results: Path, vendor: str, gpu_index: int | None,
              concurrencies: list[int], start_timeout: int,
              delete_weights: bool, dry_run: bool) -> list[dict]:
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
    # Weights are KEPT by default so re-runs skip the re-download. Only remove
    # them when --delete-weights is passed (old behavior).
    if not dry_run and delete_weights:
        removed = delete_weights(model["id"])
        for c in cells:
            c["weights_removed"] = removed
        print(f"[model {model_key}] weights removed: {removed}")
    else:
        for c in cells:
            c["weights_removed"] = False
        print(f"[model {model_key}] weights kept (re-run skips download; "
              f"run clean.sh to free ~{model.get('weights_gb', '?')} GB)")
    return cells


def main() -> int:
    args = parse_args()
    doc = yaml.safe_load(Path(args.config).read_text())
    workload, common, models = doc["workload"], doc["common_server"], doc["models"]
    configs = doc.get("configs", {})
    configs_by_name = {name: {**cfg, "name": name} for name, cfg in configs.items()}

    if args.quick:
        # Smoke mode: single pass, no external per-level warmup (the
        # in-bench --num-warmups still applies).
        model_keys, names, concurrencies = ["M1"], ["baseline", "kv-fp8"], [1, 8]
        workload["num_passes"] = 1
        workload["warmup_max_passes"] = 0
    else:
        model_keys, names, concurrencies = (
            list(models.keys()), None,
            list(workload.get("concurrency_levels", [1, 4, 8, 16])))
    if args.models:
        model_keys = _expand_model_keys(args.models, list(models.keys()))
    if args.concurrency:
        concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    if args.configs:
        names = [c.strip() for c in args.configs.split(",") if c.strip()]

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    # ── per-run sub-directory (created after GPU selection so the name
    #    includes the actual GPU model; repeated runs never collide).     ──
    gpu_index: int | None = None
    gpu: str | None = None
    run_dir = results
    if not args.dry_run:
        gpu_index = select_gpu(args.vendor, args.gpu_index)
        gpu = gpu_name(args.vendor, gpu_index)
        ts = os.environ.get("RUN_ID") or datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        base = f"{ts}_{slugify(gpu)}"
        run_dir, n = results / base, 2
        while run_dir.exists():
            run_dir, n = results / f"{base}-{n}", n + 1
        run_dir.mkdir(parents=True)
        print(f"[run_matrix] run dir: {run_dir}")
        try:
            env_info = collect_environment(args.vendor, gpu_index, gpu)
            (run_dir / "environment.json").write_text(
                json.dumps(env_info, indent=2) + "\n")
            print(f"[run_matrix] environment → {run_dir / 'environment.json'}")
        except Exception as e:  # environment is best-effort, never fatal
            print(f"WARN: environment collection failed: {e}", file=sys.stderr)

    all_cells: list[dict] = []
    for mk in model_keys:
        if mk not in models:
            print(f"WARN: unknown model {mk}, skipping", file=sys.stderr)
            continue
        model = models[mk]
        sel = names or model.get("configs", list(configs_by_name.keys()))
        cfgs = [configs_by_name[n] for n in sel if n in configs_by_name]
        all_cells.extend(run_model(mk, model, cfgs, workload, common,
                                   run_dir,
                                   args.vendor, gpu_index, concurrencies,
                                   args.start_timeout, args.delete_weights,
                                   args.dry_run))

    # attach GPU name to every cell for report.py
    for c in all_cells:
        if gpu:
            c["gpu"] = gpu
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "workload": workload,
        "common_server": common,
        "models": {mk: models[mk] for mk in model_keys if mk in models},
        "cells": all_cells,
    }
    (run_dir / "cells.json").write_text(json.dumps(manifest, indent=2))
    if not args.dry_run:
        # pointer for entrypoint.sh / bench.sh (basename only — each side
        # joins with its own $RESULTS path)
        (results / ".latest").write_text(run_dir.name + "\n")

    def _cell_failed(c: dict) -> bool:
        """Cell failed if its status is failed OR any concurrency level
        failed (server can be healthy while a bench run crashes)."""
        if c["status"] == "failed":
            return True
        return any(l.get("status") == "failed"
                   for l in c.get("concurrency_results", {}).values())

    n_ok = sum(1 for c in all_cells if c["status"] == "ok" and not _cell_failed(c))
    n_skip = sum(1 for c in all_cells if c["status"].startswith("skipped"))
    n_fail = sum(1 for c in all_cells if _cell_failed(c))
    print(f"\n[run_matrix] {n_ok} ok, {n_skip} skipped, {n_fail} failed "
          f"of {len(all_cells)} cells → {run_dir / 'cells.json'}")
    print(f"[run_matrix] RUN_DIR={run_dir}")
    return 1 if n_fail and not args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())