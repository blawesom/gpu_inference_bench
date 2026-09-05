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
  * For Intel the metrics come from two sources (see
    docs/intel-telemetry-eval.md):
      1. the xe driver's sysfs — frequency (xe ≥ 6.9) plus temperature and
         power via the xe hwmon device (recent kernels). Docker's default
         read-only /sys is not masked for these paths, so no extra mounts,
         devices, or capabilities are needed.
      2. xpu-smi — utilization + memory (always), and temperature/power/
         frequency as a fallback when the driver's sysfs lacks them.
         intel_gpu_top is i915-only and remains a last-resort util fallback.
  * Power is best-effort: if no source has a power field the aggregate
    carries null (the report renders "n/a") rather than failing.
  * Sampling is best-effort and never fatal: any tool error yields a null
    sample, and aggregate() reports what it has.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import threading
import time
from typing import Optional

INTERVAL_S = 1.0  # 1 Hz

# xe-driver sysfs roots (module constants so tests can point them at a
# fake tree). Docker mounts the host /sys read-only and does NOT mask
# these paths (runc only masks /sys/firmware and
# /sys/devices/virtual/powercap), so they are readable in-container with
# no extra mounts or capabilities.
SYSFS_DRM_ROOT = "/sys/class/drm"
SYSFS_HWMON_ROOT = "/sys/class/hwmon"


def _read_sysfs_number(path: str) -> Optional[int]:
    """Read an integer from a sysfs attribute file; None on any failure."""
    try:
        with open(path, errors="replace") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _bdf_key(bdf: str) -> tuple:
    """Sort key for a PCI BDF ('0000:0b:00.0'); unparseable sorts last."""
    try:
        dom, bus, dev_fn = bdf.split(":")
        dev, fn = dev_fn.split(".")
        return (int(dom, 16), int(bus, 16), int(dev, 16), int(fn, 16))
    except ValueError:
        return (9999, 9999, 9999, 9999)


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
        # Intel: cache the xe sysfs discovery (paths are stable for the
        # container's lifetime), the xpu-smi metric list that worked, and
        # which source produced each metric.
        self._xe_cache: Optional[dict] = None
        self._xpu_smi_metrics: Optional[str] = None
        self._intel_metric_sources: dict = {}

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
        temp = [s["temp_c"] for s in samples if s.get("temp_c") is not None]
        freq = [s["freq_mhz"] for s in samples if s.get("freq_mhz") is not None]
        return {
            "n_samples": len(samples),
            "duration_s": round(samples[-1]["t"] - samples[0]["t"], 2),
            "mem_peak_gb": round(max(mem), 2) if mem else None,
            "mem_avg_gb": round(sum(mem) / len(mem), 2) if mem else None,
            "util_avg_pct": round(sum(util) / len(util), 1) if util else None,
            "util_peak_pct": round(max(util), 1) if util else None,
            "power_avg_w": round(sum(power) / len(power), 1) if power else None,
            "power_peak_w": round(max(power), 1) if power else None,
            # Additive fields (report.md columns unchanged; the report.json
            # 'telemetry' object simply carries them along).
            "temp_avg_c": round(sum(temp) / len(temp), 1) if temp else None,
            "temp_peak_c": round(max(temp), 1) if temp else None,
            "freq_avg_mhz": round(sum(freq) / len(freq), 1) if freq else None,
            "freq_peak_mhz": round(max(freq), 1) if freq else None,
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

    # ── Intel: xe sysfs discovery ─────────────────────────────────────────
    @staticmethod
    def _find_freq_attrs(dev_dir: str) -> list[str]:
        """All cur_freq_mhz files under a card device dir.

        Covers the known xe layouts: single-GT (attribute directly on the
        card device) and per-GT (gt0/ or gt/gt0/ — the exact per-GT layout
        varies across xe releases, so we glob rather than hardcode)."""
        hits = [os.path.join(dev_dir, "cur_freq_mhz")]
        for pattern in (os.path.join(dev_dir, "gt*", "cur_freq_mhz"),
                        os.path.join(dev_dir, "gt", "gt*", "cur_freq_mhz")):
            hits += sorted(glob.glob(pattern))
        return [p for p in hits if os.path.isfile(p)]

    def _discover_xe_sysfs(self) -> dict:
        """Locate the xe card's sysfs telemetry sources (cached).

        Returns {'freq_paths': [cur_freq_mhz, ...], 'hwmon_dir': str|None}.

        xe cards are distinguished from i915 cards by the *_freq_mhz
        attributes (i915 does not expose them). Single-card is the common
        case (bench runs are single-GPU). With multiple xe cards we map the
        sampler's gpu_index onto cards sorted by PCI BDF — Level Zero
        enumerates devices in PCI order — and say so (heuristic).
        """
        if self._xe_cache is not None:
            return self._xe_cache
        cards: list[dict] = []
        for card in sorted(glob.glob(os.path.join(SYSFS_DRM_ROOT, "card*"))):
            dev = os.path.join(card, "device")
            if not os.path.isdir(dev):
                continue
            freq_paths = self._find_freq_attrs(dev)
            if freq_paths:
                bdf = os.path.basename(os.path.realpath(dev))
                cards.append({"dev": dev, "bdf": bdf,
                              "freq_paths": freq_paths})
        cache: dict = {"freq_paths": [], "hwmon_dir": None}
        if cards:
            cards.sort(key=lambda c: _bdf_key(c["bdf"]))
            if len(cards) > 1:
                chosen = (cards[self.gpu_index]
                          if self.gpu_index is not None
                          and 0 <= self.gpu_index < len(cards)
                          else cards[0])
                print(f"[telemetry] intel: {len(cards)} xe cards visible "
                      f"({[c['bdf'] for c in cards]}); using {chosen['bdf']} "
                      f"for index {self.gpu_index} (PCI-order heuristic)")
            else:
                chosen = cards[0]
            cache["freq_paths"] = chosen["freq_paths"]
            # The xe hwmon device (temperature/power) symlinks 'device' to
            # the same PCI dir as the card; prefer an exact match, fall
            # back to any xe-named hwmon (single-GPU systems).
            pci_real = os.path.realpath(chosen["dev"])
            hwmon_xe: list[str] = []
            for h in sorted(glob.glob(
                    os.path.join(SYSFS_HWMON_ROOT, "hwmon*"))):
                try:
                    with open(os.path.join(h, "name"), errors="replace") as f:
                        name = f.read().strip()
                except OSError:
                    continue
                if "xe" not in name.lower():
                    continue
                dev_link = os.path.join(h, "device")
                if os.path.exists(dev_link) \
                        and os.path.realpath(dev_link) == pci_real:
                    cache["hwmon_dir"] = h
                    break
                hwmon_xe.append(h)
            if cache["hwmon_dir"] is None and hwmon_xe:
                cache["hwmon_dir"] = hwmon_xe[0]
        self._xe_cache = cache
        return cache

    def _sysfs_freq(self) -> Optional[float]:
        """GPU frequency (MHz) from xe sysfs; max across GTs (the compute
        GT is the one that ramps). None if the driver/kernel lacks it."""
        paths = self._discover_xe_sysfs()["freq_paths"]
        if not paths:
            return None
        vals = [v for v in (_read_sysfs_number(p) for p in paths)
                if v is not None]
        if not vals:
            return None
        f = max(vals)
        return float(f) if 100 <= f <= 10000 else None

    def _sysfs_hwmon(self) -> tuple:
        """(temp °C, power W) from the xe hwmon device.

        hwmon convention: temp*_input in millidegrees C, power*_input in
        milliwatts. Both None on kernels without the xe hwmon driver."""
        h = self._discover_xe_sysfs()["hwmon_dir"]
        if not h:
            return None, None
        temp = None
        for attr in ("temp1_input", "temp2_input"):
            v = _read_sysfs_number(os.path.join(h, attr))
            if v is not None:
                t = v / 1000.0
                if 5.0 <= t <= 125.0:  # out-of-range = sensor not ready
                    temp = round(t, 1)
                break
        power = None
        for attr in ("power1_input", "power1_average"):
            v = _read_sysfs_number(os.path.join(h, attr))
            if v is not None:
                w = v / 1000.0
                if 0.0 <= w <= 1000.0:
                    power = round(w, 1)
                break
        return temp, power

    # ── Intel: sampling ────────────────────────────────────────────────────
    def _sample_intel(self) -> Optional[dict]:
        # Three sources, in order of preference per metric:
        #  1. xe sysfs — zero deps (freq on xe ≥ 6.9; temp/power need the
        #     xe hwmon driver, recent kernels).
        #  2. xpu-smi — util + mem (always), temp/power/freq as fallback
        #     when sysfs lacks them.
        #  3. intel_gpu_top — i915-only last resort for utilization.
        result = {"mem_used_gb": None, "util_pct": None, "power_w": None,
                  "temp_c": None, "freq_mhz": None}
        src: dict = {}
        freq = self._sysfs_freq()
        if freq is not None:
            result["freq_mhz"] = freq
            src["frequency"] = "xe-sysfs"
        temp, power = self._sysfs_hwmon()
        if temp is not None:
            result["temp_c"] = temp
            src["temperature"] = "xe-sysfs"
        if power is not None:
            result["power_w"] = power
            src["power"] = "xe-sysfs"
        self._xpu_smi_fill(result, src)
        if result["util_pct"] is None:
            env = {k: v for k, v in os.environ.items()
                   if k not in ("ONEAPI_DEVICE_SELECTOR",)}
            out = _run(["intel_gpu_top", "-l", "-s", "1"], env=env,
                       timeout=10.0)
            if out:
                m = re.search(r"GPU Util:\s*(\d+(?:\.\d+)?)%", out)
                if m:
                    result["util_pct"] = float(m.group(1))
                    src["utilization"] = "intel_gpu_top"
        self._intel_metric_sources.update(src)
        if any(v is not None for v in result.values()):
            return result
        return None

    _XPU_SMI_FULL_METRICS = ("gpu_utilization,mem_used,temperature,"
                             "power,gpu_frequency")
    _XPU_SMI_LEGACY_METRICS = "gpu_utilization,mem_used"

    def _xpu_smi_fill(self, result: dict, src: dict) -> None:
        """Fill whatever xpu-smi reports, never overwriting values sysfs
        already produced. Best-effort: any tool error leaves fields null."""
        idx = str(self.gpu_index if self.gpu_index is not None else 0)
        env = {k: v for k, v in os.environ.items()
               if k not in ("ONEAPI_DEVICE_SELECTOR",)}
        # Extended metric set first; older builds can reject unknown metric
        # names for the whole dump, so fall back to the legacy pair (and
        # remember which list worked).
        metrics = (self._xpu_smi_metrics or self._XPU_SMI_FULL_METRICS)
        out = _run(["xpu-smi", "dump", "-d", idx, "-m", metrics],
                   env=env, timeout=10.0)
        if not out and metrics != self._XPU_SMI_LEGACY_METRICS:
            out = _run(["xpu-smi", "dump", "-d", idx, "-m",
                        self._XPU_SMI_LEGACY_METRICS], env=env, timeout=10.0)
            if out:
                self._xpu_smi_metrics = self._XPU_SMI_LEGACY_METRICS
        elif out:
            self._xpu_smi_metrics = metrics
        if not out:
            return
        m = re.search(r"gpu_utilization[\"':\s]+(\d+(?:\.\d+)?)", out)
        if m and result["util_pct"] is None:
            result["util_pct"] = float(m.group(1))
            src["utilization"] = "xpu-smi"
        m_mem = re.search(r"mem_used[\"':\s]+(\d+(?:\.\d+)?)", out)
        if m_mem and result["mem_used_gb"] is None:
            v = float(m_mem.group(1))
            # Unit heuristic: xpu-smi reports bytes for large
            # values, MiB in some builds — distinguish by magnitude
            # (a fully used 16 GB card is ~1.7e10 bytes vs ~1.6e4 MiB).
            result["mem_used_gb"] = round(v / 1024 ** 3, 2) if v >= 10 ** 8 \
                else round(v / 1024.0, 2)
            src["memory"] = "xpu-smi"
        # Fallback metrics (only used when sysfs did not provide them).
        if result["temp_c"] is None:
            m_t = re.search(
                r"(?<![\w])(?:gpu_)?temperature[\"':\s]+(-?\d+(?:\.\d+)?)",
                out)
            if m_t:
                v = float(m_t.group(1))
                if 5.0 <= v <= 125.0:
                    result["temp_c"] = v
                    src["temperature"] = "xpu-smi"
        if result["power_w"] is None:
            m_p = re.search(
                r"(?<![\w])(?:gpu_)?power[\"':\s]+(-?\d+(?:\.\d+)?)", out)
            if m_p:
                v = float(m_p.group(1))
                if 0.1 <= v <= 1000.0:
                    result["power_w"] = v
                    src["power"] = "xpu-smi"
        if result["freq_mhz"] is None:
            m_f = re.search(
                r"(?<![\w])(?:gpu_)?frequency[\"':\s]+(\d+(?:\.\d+)?)", out)
            if m_f:
                v = float(m_f.group(1))
                if 100.0 <= v <= 10000.0:
                    result["freq_mhz"] = v
                    src["frequency"] = "xpu-smi"

    # ── environment probe (used by run_matrix.collect_environment) ────────
    def probe(self) -> Optional[dict]:
        """One synchronous sample (best-effort). Used at run start to
        record which telemetry sources this host actually provides."""
        return self._sample()

    def metric_sources(self) -> dict:
        """{metric: source} recorded from the latest samples
        ('xe-sysfs' | 'xpu-smi' | 'intel_gpu_top')."""
        return dict(self._intel_metric_sources)

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