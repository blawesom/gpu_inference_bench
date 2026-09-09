#!/usr/bin/env python3
"""Intel GPU telemetry spike probe (2026-09-08 review, intel-telemetry-eval §7).

One-shot, READ-ONLY diagnostic that answers the single question blocking a
definitive Intel energy number: *which power source does this box actually
expose, from where I will run the bench (host vs. the vLLM XPU container)?*

It reuses the production sampler's own discovery/parsing logic (imported from
``telemetry.py``), so a "power: OK" verdict here means the real run will get
power too — it is not a re-implementation that could disagree.

Run it TWICE for a complete picture:
  1. On the B70 host:           python3 container/intel_telemetry_probe.py
  2. Inside the vLLM XPU image: python3 intel_telemetry_probe.py
     (copy this file into the image, or mount it read-only)

Output: a JSON report on stdout (and, with --out, a file) plus a short human
summary on stderr. Exit code is always 0 — this is a probe, not a gate.

Nothing here writes to the GPU, changes drivers, or installs anything.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import telemetry as T
except ImportError:  # probe should still report what it can
    T = None

DRM_ROOT = os.environ.get("INTEL_PROBE_DRM_ROOT", "/sys/class/drm")
HWMON_ROOT = os.environ.get("INTEL_PROBE_HWMON_ROOT", "/sys/class/hwmon")


# ── small helpers ───────────────────────────────────────────────────────────
def _run(cmd, timeout=15.0):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception as e:  # noqa: BLE001 - probe must never crash
        return f"[probe error: {e}]"


def _run_both(cmd, timeout=15.0):
    """(stdout, stderr) — for diagnosing why a tool produced no output."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:  # noqa: BLE001
        return "", f"[probe error: {e}]"


def _read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _in_container() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        cg = _read("/proc/1/cgroup") or ""
        return ("docker" in cg or "containerd" in cg or "libpod" in cg)
    except OSError:
        return False


def _xe_cards():
    """[(card, dev, bdf, freq_paths)] for every xe card (i915 has no
    *_freq_mhz). Mirrors TelemetrySampler._find_freq_attrs."""
    cards = []
    for card in sorted(glob.glob(os.path.join(DRM_ROOT, "card*"))):
        dev = os.path.join(card, "device")
        if not os.path.isdir(dev):
            continue
        freq_paths = (T.TelemetrySampler._find_freq_attrs(dev)
                      if T is not None else [])
        if not freq_paths:
            continue
        bdf = os.path.basename(os.path.realpath(dev))
        cards.append({"card": card, "dev": dev, "bdf": bdf,
                      "freq_paths": freq_paths})
    return cards


def _hwmon_devices():
    out = []
    for h in sorted(glob.glob(os.path.join(HWMON_ROOT, "hwmon*"))):
        name = _read(os.path.join(h, "name"))
        attrs = sorted(os.path.basename(p) for p in glob.glob(h + "/*")
                       if os.path.isfile(p))
        out.append({"dir": h, "name": name,
                    "is_xe": bool(name) and "xe" in name.lower(),
                    "has_temp": any(a.startswith("temp") and a.endswith("_input")
                                    for a in attrs),
                    "has_power": any(a.startswith("power") and
                                     a.endswith(("_input", "_average"))
                                     for a in attrs)})
    return out


# ── individual checks ────────────────────────────────────────────────────────
def probe_system():
    xv = _run(["modinfo", "-F", "version", "xe"])
    m = re.search(r"(\S+)", xv)
    return {
        "kernel": _run(["uname", "-r"]),
        "xe_module_version": m.group(1) if m else None,
        "in_container": _in_container(),
        "rapl_visible": bool(glob.glob("/sys/class/powercap/intel-rapl*")),
    }


def probe_sysfs():
    cards = _xe_cards()
    hwmon = _hwmon_devices()
    xe = []
    for c in cards:
        dev = c["dev"]
        entry = {
            "bdf": c["bdf"], "card": c["card"],
            "freq_paths": c["freq_paths"],
            "freq_values_mhz": [_read(p) for p in c["freq_paths"]],
            "render_nodes": sorted(
                os.path.basename(p) for p in glob.glob(os.path.join(dev, "renderD*"))),
            "hwmon": None, "hwmon_temp_c": None, "hwmon_power_w": None,
        }
        # Bind an xe hwmon to this card via its device symlink.
        pci_real = os.path.realpath(dev)
        match = None
        for h in hwmon:
            if not h["is_xe"]:
                continue
            dl = os.path.join(h["dir"], "device")
            if os.path.exists(dl) and os.path.realpath(dl) == pci_real:
                match = h
                break
        if match:
            entry["hwmon"] = match["dir"]
            for a in ("temp1_input", "temp2_input"):
                v = _read(os.path.join(match["dir"], a))
                if v is not None:
                    try:
                        entry["hwmon_temp_c"] = round(int(v) / 1000.0, 1)
                    except ValueError:
                        pass
                    break
            for a in ("power1_input", "power1_average"):
                v = _read(os.path.join(match["dir"], a))
                if v is not None:
                    try:
                        entry["hwmon_power_w"] = round(int(v) / 1000.0, 1)
                    except ValueError:
                        pass
                    break
        xe.append(entry)
    # Fallback: even when no card is discovered via *_freq_mhz (as on some
    # Xe2/BM-G kernels), the xe-named hwmon may still expose temperature.
    # Report it so we know temp is available via sysfs regardless of freq.
    xe_hwmon_fallback = None
    if not xe:
        for h in hwmon:
            if not h["is_xe"]:
                continue
            t = None
            for a in ("temp1_input", "temp2_input"):
                v = _read(os.path.join(h["dir"], a))
                if v is not None:
                    try:
                        t = round(int(v) / 1000.0, 1)
                    except ValueError:
                        pass
                    break
            p = None
            for a in ("power1_input", "power1_average"):
                v = _read(os.path.join(h["dir"], a))
                if v is not None:
                    try:
                        p = round(int(v) / 1000.0, 1)
                    except ValueError:
                        pass
                    break
            xe_hwmon_fallback = {"hwmon": h["dir"], "temp_c": t, "power_w": p}
            break
    return {"xe_cards": xe, "xe_hwmon_fallback": xe_hwmon_fallback,
            "hwmon_devices": hwmon}


def probe_tools():
    detail = {}
    xpu_smi = shutil.which("xpu-smi")
    zeinfo = shutil.which("zeinfo")
    igt = shutil.which("intel_gpu_top")
    if xpu_smi:
        ver_out, _ = _run_both(["xpu-smi", "--version"])
        d = {"path": xpu_smi,
             "version": ver_out.splitlines()[0] if ver_out else None}
        # Device enumeration (does xpu-smi see a device at all?).
        d["discovery"] = _run(["xpu-smi", "discovery"], timeout=20.0)[:1200] or None
        # Full metric set — capture stderr too: a null dump with an error on
        # stderr usually means a permissions/group problem, not "no metric".
        ext = (["xpu-smi", "dump", "-d", "0", "-m",
                "gpu_utilization,mem_used,temperature,power,gpu_frequency"])
        d["dump_d0_extended"], d["dump_d0_extended_stderr"] = _run_both(ext, 25.0)
        d["dump_d0_extended"] = d["dump_d0_extended"] or None
        d["dump_d0_extended_stderr"] = (d["dump_d0_extended_stderr"] or "")[:600] or None
        # Power alone — isolates whether the 'power' metric is the problem.
        d["dump_d0_power"], d["dump_d0_power_stderr"] = _run_both(
            ["xpu-smi", "dump", "-d", "0", "-m", "power"], 25.0)
        d["dump_d0_power"] = d["dump_d0_power"] or None
        d["dump_d0_power_stderr"] = (d["dump_d0_power_stderr"] or "")[:600] or None
        detail["xpu-smi"] = d
    if zeinfo:
        detail["zeinfo"] = {"path": zeinfo,
                            "version": (_run(["zeinfo", "-v"]).splitlines()[0]
                                        if _run(["zeinfo", "-v"]) else None)}
    if igt:
        detail["intel_gpu_top"] = {"path": igt}
    return {"xpu-smi": bool(xpu_smi), "zeinfo": bool(zeinfo),
            "intel_gpu_top": bool(igt), "detail": detail}


def probe_live_sampler(gpu_index=None, seconds=5.0):
    """Run the REAL production sampler briefly. Ground truth: if it sees
    power here, the bench will too."""
    if T is None:
        return {"error": "telemetry module not importable"}
    out = {"sampler": "TelemetrySampler('intel', ...)", "seconds": seconds}
    try:
        s = T.TelemetrySampler("intel", gpu_index)
        s.start()
        time.sleep(seconds)
        samples = s.stop()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"sampler raised: {e}"
        return out
    n = len(samples)
    out["n_samples"] = n
    if not samples:
        out["verdict"] = "NO SAMPLES — every source returned nothing"
        return out
    out["metric_sample_counts"] = {
        m: sum(1 for x in samples if x.get(m) is not None)
        for m in ("power_w", "temp_c", "freq_mhz", "util_pct", "mem_used_gb")}
    out["metric_sources"] = s.metric_sources()
    for m in ("power_w", "freq_mhz", "temp_c"):
        vals = [x[m] for x in samples if x.get(m) is not None]
        out[f"{m}_min_max"] = [min(vals), max(vals)] if vals else None
    out["verdict"] = ("power OK via " + s.metric_sources().get("power", "?")
                      if out["metric_sample_counts"]["power_w"]
                      else "NO POWER from any source")
    return out


# ── verdict ─────────────────────────────────────────────────────────────────
def _xpu_smi_power_works(tools: dict) -> bool:
    """True if a power-only (or extended) xpu-smi dump returned a number.
    Prefers the dedicated 'power' dump, falls back to the extended dump."""
    det = (tools.get("detail") or {}).get("xpu-smi") or {}
    for key in ("dump_d0_power", "dump_d0_extended"):
        dump = det.get(key)
        if not dump:
            continue
        m = re.search(r"(?<![\w])power[\"':\s]+(\d+(?:\.\d+)?)", dump, re.I)
        if m and float(m.group(1)) > 0:
            return True
    return False


def make_verdict(system, sysfs, tools, live):
    xe = sysfs.get("xe_cards", [])
    fb = sysfs.get("xe_hwmon_fallback") or {}
    hwmon_power = any(c.get("hwmon_power_w") is not None for c in xe) or \
        fb.get("power_w") is not None
    hwmon_temp = any(c.get("hwmon_temp_c") is not None for c in xe) or \
        fb.get("temp_c") is not None
    sysfs_freq = any(any(v is not None for v in (c.get("freq_values_mhz") or []))
                     for c in xe)
    xpu_power = bool(tools.get("xpu-smi")) and _xpu_smi_power_works(tools)
    msrc = (live.get("metric_sources") or {}) if live else {}
    live_power = msrc.get("power") is not None

    power_sources = []
    if hwmon_power:
        power_sources.append("xe-sysfs-hwmon")
    if xpu_power:
        power_sources.append("xpu-smi")
    if live_power and msrc.get("power") not in power_sources:
        power_sources.append(msrc.get("power"))
    power_available = bool(power_sources)

    # Diagnostic: why did xpu-smi's dump come back empty?
    xs = (tools.get("detail") or {}).get("xpu-smi") or {}
    xpu_dump_empty = bool(xs) and not xs.get("dump_d0_power") \
        and not xs.get("dump_d0_extended")
    xpu_stderr = xs.get("dump_d0_power_stderr") or \
        xs.get("dump_d0_extended_stderr")

    v = {
        "power_available": power_available,
        "power_sources": power_sources,
        "temperature": ("xe-sysfs-hwmon" if hwmon_temp
                        else (msrc.get("temperature")
                              if msrc.get("temperature") not in (None, "none")
                              else None)),
        "frequency": ("xe-sysfs" if sysfs_freq
                      else (msrc.get("frequency")
                            if msrc.get("frequency") not in (None, "none")
                            else None)),
        "utilization": msrc.get("utilization"),
        "memory": msrc.get("memory"),
        "hwmon_power_w_sampled": next(
            (c.get("hwmon_power_w") for c in xe
             if c.get("hwmon_power_w") is not None), None),
        "hwmon_temp_c_sampled": next(
            (c.get("hwmon_temp_c") for c in xe
             if c.get("hwmon_temp_c") is not None), None),
    }
    if power_available:
        v["recommendation"] = (
            f"POWER OK — available via {', '.join(power_sources)}. Tier 1 is "
            "sufficient; run the T0 Intel reference as planned and the "
            "cross-system comparison can drop the assumed-TDP floor.")
    else:
        v["recommendation"] = (
            "POWER NOT AVAILABLE on this kernel/container. Tier 2 (opt-in "
            "wrapper image that adds xpu-smi) is required before an Intel "
            "energy ranking is meaningful; until then the comparison must "
            "keep the 230 W assumed-TDP floor and a 'no GPU telemetry' "
            "caveat (no definitive energy ranking).")
    return v


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description="Intel GPU telemetry spike probe (read-only).")
    p.add_argument("--gpu-index", type=int, default=None)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--out", default=None,
                   help="also write the JSON report to this path")
    p.add_argument("--no-live", action="store_true",
                   help="skip the live sampler (static discovery only)")
    args = p.parse_args()

    system = probe_system()
    sysfs = probe_sysfs()
    tools = probe_tools()
    live = ({} if args.no_live else
            probe_live_sampler(args.gpu_index, args.seconds))
    if args.no_live:
        live = {"skipped": True}

    report = {
        "probe": "intel_telemetry",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": system,
        "sysfs": sysfs,
        "tools": tools,
        "live_sampler": live,
        "verdict": make_verdict(system, sysfs, tools, live),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)

    v = report["verdict"]
    s = report["system"]
    print("\n=== INTEL TELEMETRY PROBE SUMMARY ===", file=sys.stderr)
    print(f"  where      : {'container' if s['in_container'] else 'host'}"
          f"   kernel {s['kernel']}   xe {s['xe_module_version']}",
          file=sys.stderr)
    for c in sysfs.get("xe_cards", []):
        print(f"  xe {c['bdf']}: freq={c['freq_values_mhz']}  "
              f"hwmon={c.get('hwmon')}  temp={c.get('hwmon_temp_c')}C  "
              f"power={c.get('hwmon_power_w')}W", file=sys.stderr)
    t = report["tools"]
    xs = t.get("detail", {}).get("xpu-smi") or {}
    xs_st = xs.get("dump_d0_power_stderr") or xs.get("dump_d0_extended_stderr")
    xs_dis = xs.get("discovery")
    xs_ver = xs.get("version")
    print(f"  tools      : xpu-smi={t['xpu-smi']} ver={xs_ver}  "
          f"intel_gpu_top={t['intel_gpu_top']}", file=sys.stderr)
    if xs_st:
        print(f"  xpu-smi    : dump stderr = {xs_st[:300]}", file=sys.stderr)
    if xs_dis:
        print(f"  xpu-smi    : discovery = {xs_dis[:300]}", file=sys.stderr)
    fb = report["sysfs"].get("xe_hwmon_fallback")
    if fb:
        print(f"  xe-fallback: hwmon={fb.get('hwmon')}  temp={fb.get('temp_c')}C"
              f"  power={fb.get('power_w')}W", file=sys.stderr)
    power_str = ('AVAILABLE via ' + ','.join(v['power_sources'])
                 if v['power_available'] else 'NOT AVAILABLE')
    print(f"  power      : {power_str}", file=sys.stderr)
    print(f"  verdict    : {v['recommendation']}", file=sys.stderr)
    print("=====================================", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())