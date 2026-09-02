#!/usr/bin/env python3
"""telemetry.py — 1 Hz GPU sampler (best-effort, per-vendor).

Used by run_matrix.py to capture GPU memory / utilization / power while a
`vllm bench serve` run is in flight. Each (cell, concurrency) level gets its
own sample window, so the report can show per-level GPU stats.

Design notes:
  * The sampler runs INSIDE the vllm container, where the vendor CLI is present.
  * For AMD the physical GPU index must be sampled from the *unfiltered*
    rocm-smi table (rocm-smi ignores HIP_VISIBLE_DEVICES), so we strip that
    var from the sampler's environment and select the row by index.
  * Power is best-effort: if the vendor tool has no power field the aggregate
    carries null (the report renders "n/a") rather than failing.
  * Sampling is best-effort and never fatal: any tool error yields a null
    sample, and aggregate() reports what it has.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Optional

INTERVAL_S = 1.0  # 1 Hz


def _run(cmd: list[str], env: dict | None = None, timeout: float = 10.0) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        if r.returncode == 0:
            return r.stdout
    except (OSError, subprocess.SubprocessError):
        pass
    return None


class TelemetrySampler:
    def __init__(self, vendor: str, gpu_index: Optional[int] = None):
        self.vendor = (vendor or "auto").lower()
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: list[dict] = []
        self._lock = threading.Lock()
        # AMD: cache total VRAM (bytes) per physical GPU once (unfiltered).
        self._amd_total_bytes: dict[int, int] = {}
        if self.vendor == "amd":
            self._query_amd_totals()

    # ── public API ──────────────────────────────────────────────────────────
    def start(self) -> list[dict]:
        """Begin sampling in a background thread; returns the live sample list."""
        with self._lock:
            self._samples = []
            self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self._samples

    def stop(self) -> list[dict]:
        """Stop sampling; return the collected samples."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            return list(self._samples)

    def aggregate(self, samples: list[dict]) -> Optional[dict]:
        """Collapse samples into peak/avg metrics. None if no usable samples."""
        if not samples:
            return None
        mem = [s["mem_used_gb"] for s in samples
               if s.get("mem_used_gb") is not None]
        util = [s["util_pct"] for s in samples if s.get("util_pct") is not None]
        power = [s["power_w"] for s in samples if s.get("power_w") is not None]
        return {
            "n_samples": len(samples),
            "duration_s": round(samples[-1]["t"] - samples[0]["t"], 2),
            "mem_peak_gb": round(max(mem), 2) if mem else None,
            "mem_avg_gb": round(sum(mem) / len(mem), 2) if mem else None,
            "util_avg_pct": round(sum(util) / len(util), 1) if util else None,
            "util_peak_pct": round(max(util), 1) if util else None,
            "power_avg_w": round(sum(power) / len(power), 1) if power else None,
            "power_peak_w": round(max(power), 1) if power else None,
        }

    # ── sampling loop ───────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            sample = self._sample()
            if sample is not None:
                sample["t"] = time.time()
                with self._lock:
                    self._samples.append(sample)
            self._stop.wait(max(0.0, INTERVAL_S - (time.time() - t0)))

    def _sample(self) -> Optional[dict]:
        if self.vendor == "nvidia":
            return self._sample_nvidia()
        if self.vendor == "amd":
            return self._sample_amd()
        if self.vendor == "intel":
            return self._sample_intel()
        return None

    # ── vendor samplers ─────────────────────────────────────────────────────
    def _sample_nvidia(self) -> Optional[dict]:
        # -i 0 is the only visible GPU when bench.sh passes --gpus device=N;
        # gpu_index selects when multiple are visible (unfiltered nvidia-smi).
        idx = str(self.gpu_index if self.gpu_index is not None else 0)
        out = _run(["nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits", "-i", idx])
        if not out:
            return None
        parts = [p.strip() for p in out.split(",")]
        try:
            mem_mib = float(parts[0])
            util = float(parts[1]) if len(parts) > 1 and parts[1] else None
            power = float(parts[2]) if len(parts) > 2 and parts[2] else None
        except (ValueError, IndexError):
            return None
        return {"mem_used_gb": round(mem_mib / 1024.0, 2),
                "util_pct": util,
                "power_w": (power if power and power > 0 else None)}

    def _query_amd_totals(self) -> None:
        # unfiltered, so physical indices line up with the full table
        env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
        out = _run(["rocm-smi", "--showmeminfo", "vram"], env=env, timeout=20.0)
        if not out:
            return
        for m in re.finditer(r"GPU\[(\d+)\].*?VRAM Total Memory \(B\): (\d+)",
                             out):
            self._amd_total_bytes[int(m.group(1))] = int(m.group(2))

    def _sample_amd(self) -> Optional[dict]:
        # unfiltered concise table → Power (W), VRAM%, GPU% for the physical GPU
        env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
        out = _run(["rocm-smi"], env=env, timeout=20.0)
        if not out:
            return None
        for line in out.splitlines():
            s = line.strip()
            if not s or not s[0].isdigit():
                continue
            parts = s.split()
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            if self.gpu_index is not None and idx != self.gpu_index:
                continue
            power = None
            for p in parts:
                if re.match(r"^\d+(\.\d+)?W$", p):
                    power = float(p[:-1])
                    break
            vram_pct = self._pct(parts[-2]) if len(parts) >= 2 else None
            gpu_pct = self._pct(parts[-1])
            mem_used_gb = None
            total = self._amd_total_bytes.get(idx)
            if total and vram_pct is not None:
                mem_used_gb = round(total / (1024 ** 3) * vram_pct / 100.0, 2)
            return {"mem_used_gb": mem_used_gb,
                    "util_pct": gpu_pct,
                    "power_w": (power if power and power > 0 else None)}
        return None

    def _sample_intel(self) -> Optional[dict]:
        # best-effort: xpu-smi preferred, intel_gpu_top fallback. No reliable
        # power field → power stays null.
        env = {k: v for k, v in os.environ.items()
               if k not in ("ONEAPI_DEVICE_SELECTOR",)}
        out = _run(["xpu-smi", "dump", "-d", str(self.gpu_index or 0),
                    "-m", "gpu_utilization,mem_used"], env=env, timeout=10.0)
        if out:
            m = re.search(r"gpu_utilization[\"':\s]+(\d+(?:\.\d+)?)", out)
            if m:
                return {"mem_used_gb": None, "util_pct": float(m.group(1)),
                        "power_w": None}
        out = _run(["intel_gpu_top", "-l", "-s", "1"], env=env, timeout=10.0)
        if out:
            m = re.search(r"GPU Util:\s*(\d+(?:\.\d+)?)%", out)
            if m:
                return {"mem_used_gb": None, "util_pct": float(m.group(1)),
                        "power_w": None}
        return None

    @staticmethod
    def _pct(tok: str) -> Optional[float]:
        if not tok:
            return None
        tok = tok.rstrip("%").strip()
        try:
            return float(tok)
        except ValueError:
            return None


if __name__ == "__main__":
    # quick self-test: sample the current vendor for ~5 s and print aggregate
    import json
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GPU_VENDOR", "auto")
    s = TelemetrySampler(v)
    live = s.start()
    time.sleep(5.0)
    samples = s.stop()
    print(json.dumps(s.aggregate(samples), indent=2))