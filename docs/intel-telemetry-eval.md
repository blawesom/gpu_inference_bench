# Intel GPU thermal & power telemetry in the bench container

> **Goal:** first-class thermal + power telemetry on the Intel (xe, Arc B70/B580) target, on par with NVIDIA/AMD (which already report power today).

**Status:** evaluation. **Tier 1 (Option A, xe-sysfs sampling) implemented**
(`telemetry.py` + `run_matrix.py` probe; report.md format unchanged, new
fields are additive in the per-level `telemetry` objects and in
`environment.json` → `telemetry_metrics`). Target-box spike (§7) still
pending.

---

## 1. Current state on Intel

| Metric | Source | Availability | In report? |
|---|---|---|---|
| Memory | `xpu-smi dump -m mem_used` | Best-effort (xpu-smi not always in image) | ✅ peak/avg |
| Utilization | `xpu-smi dump -m gpu_utilization` | Best-effort | ✅ avg/peak |
| **Power** | — | ❌ Always null — the requested metric list (`gpu_utilization,mem_used`) does not include power, and the xe sysfs path is not used | ❌ "n/a" |
| **Temperature** | — | ❌ Not collected at all | ❌ No column |
| **Frequency** | — | ❌ Not collected | ❌ No column |

### Why these gaps exist

1. **`xpu-smi` metrics list is minimal.** `telemetry.py` requests only `gpu_utilization,mem_used` (line 183). Power/temperature/frequency aren't requested.
2. **`intel_gpu_top` is i915-only** — it cannot see xe devices. Arc B70 (BMG/Xe2) runs on the `xe` driver, so this tool is irrelevant.
3. **The xe driver's own sysfs sources are unused.** The kernel exposes frequency since 6.9, and thermal/power via hwmon on recent kernels — but the container reads nothing from sysfs.
4. **The vLLM XPU image (`vllm/vllm-openai-xpu`) carries the Level Zero runtime + PyTorch XPU**, not full diagnostic tools. The repo notes xpu-smi is "not always in the image."

---

## 2. Data sources on the Intel target

| Source | Temp | Power | Freq | Util | Mem | In container | Kernel req. | Privileges |
|---|---|---|---|---|---|---|---|---|
| **xe sysfs** `*/device/*_freq_mhz` | – | – | ✅ | – | – | Read-only `/sys` (not masked) | xe ≥6.9 | none |
| **xe hwmon** `*/hwmon*/{temp1_input,power1_input}` | ✅ | ✅ | – | – | – | Read-only `/sys` (not masked) | xe + hwmon driver | none |
| **xpu-smi** (XPU-SMI-Lib) | ✅ (verify) | ✅ (verify) | ✅ (verify) | ✅ | ✅ | Tool must be present/installed | Varies by version | root (for libze access) |
| intel_gpu_top | – | – | ✅ i915 | ✅ i915 | ✅ i915 | Not in image | i915 only | none |
| Level Zero / torch.xpu | – | – | – | – | ✅ mem only | In image | – | none |
| CPU RAPL | CPU pkg W | CPU pkg W | – | – | – | `/powercap` masked by Docker | intel_pch_thermal/RAPL | bind-mount to unmask |
| CPU thermal_zone | CPU °C | – | – | – | – | Visible in container | coretemp | none |

> **Key insight:** `/sys` is mounted read-only by Docker from the host and is **not filtered** (runc's mask list includes `/sys/firmware` and `/sys/devices/virtual/powercap` but **not** `/sys/class/drm` or `/sys/class/hwmon`). The xe driver's sysfs attributes are readable from inside the container **out of the box**, with zero additional mounts or permissions.

---

## 3. Container runtime mount/permission checklist

| Check | Status | Notes |
|---|---|---|
| DRI render/card nodes passed (via bench.sh) | ✅ Done | `-v /dev/dri:/dev/dri` + `--device` per node + `--group-add` (see [intel-xpu-dev-dri.md](intel-xpu-dev-dri.md)) |
| `/sys/class/drm` visible | ✅ Default | Read-only host `/sys`; not in Docker's mask list |
| `/sys/class/hwmon` visible | ✅ Default | Not masked; contains the xe hwmon device (temp + power) |
| New cgroup devices needed? | ❌ No | sysfs reads are unprivileged |
| New capabilities needed? | ❌ No | no ptrace, no perf_event |
| xpu-smi needed in container? | See Option A/B/C | Only for Tier 2 (fallback); Tier 1 uses sysfs |
| CPU RAPL needed? | ❌ Optional | Masked by default → needs a bind mount to unmask |

---

## 4. Options for adding tools to the container

### Option A: Zero-install sysfs sampling (recommended — implement now)

Extend `telemetry.py` to read xe sysfs attributes directly. No new tools, no mounts, no permissions.

**Pros:**
- Zero dependencies — the vLLM image stays exactly as-is
- No Docker flag changes
- Available from container start (t=0, no install delay)
- Backward-compatible — degrades gracefully if a sysfs path is absent
- Kernel frequency data available since 6.9

**Cons:**
- Temperature + power via hwmon requires a recent kernel (xe hwmon merged in the 6.15 release cycle; verify on target). Older kernels will only report frequency.
- GPU utilization must still come from xpu-smi (sysfs has no utilization counter without perf_event_open)

**Coverage:** freq (≥6.9), temp + power (≥xe-hwmon kernel), util/mem via xpu-smi fallback

---

### Option B: Runtime install of xpu-smi in the entrypoint (Intel only)

Add an Intel-specific `apt-get install intel-xpu-smi` (or pip) step to `entrypoint.sh`, best-effort with a timeout.

**Pros:**
- Vendor-supported single CLI for all metrics (temp, power, freq, util, mem)
- May work on older kernels where hwmon is not yet merged
- Structured CLI output (one tool instead of multiple)

**Cons:**
- **Breaks the "pinned image" reproducibility** — the effective environment changes per run
- Requires network at benchmark start
- 1–2 min install delay on first run
- Library version mismatch risk (xpu-smi needs libze; could conflict with the image's pinned oneAPI stack)
- Network offline / air-gapped host → no telemetry

**Risk level:** medium (fragile, network-dependent, changes the effective container)

---

### Option C: Local wrapper image built on top of the official XPU image

Build a local image: `vllm/vllm-openai-xpu:0.28.0-bench` = official image + `apt install intel-xpu-smi intel-gpu-tools`.

**Pros:**
- Tools available from t=0 (no runtime install)
- Image ID recorded in `environment.json` (the schema already captures this)
- No network needed at benchmark time
- Works offline / air-gapped
- `--image` override still works; the built image is a one-time step

**Cons:**
- Requires a build step on the Intel host (~1–5 min, first run only)
- Deviates from the current "no Dockerfile" policy (but only for Intel, with an opt-in flag)
- One more moving part in the pipeline

**Recommendation:** Present as the **recommended Tier 2 option** — a `--build-xpu-tools` flag that triggers a one-off image build.

---

### Option D: Telemetry sidecar container

A second minimal container (e.g., `ubuntu:24.04` + xpu-smi) with the same DRI mounts, writing JSONL to a shared volume; `run_matrix.py` tails it.

**Pros:**
- vLLM image stays untouched
- Tools fully isolated

**Cons:**
- Highest complexity: two containers, shared volume, lifecycle management
- Duplicate DRI mounts + group-add + device flags
- Timestamp correlation across two containers
- Breaks the repo's "single container, thin orchestrator" design principle
- **Rejected as the default approach**

---

### Option E: Host-side sampler (no container changes)

`bench.sh` runs a parallel sampler on the host, writing to the results volume.

**Pros:**
- No container changes at all
- Host has the xe driver + (possibly) xpu-smi

**Cons:**
- The container already sees the host's `/sys` — so the data source is identical to Option A
- Contradicts the "tools in the container runtime" requirement
- Requires tools already installed on the host
- **Rejected** — no advantage over A

---

### Comparison summary

| Criterion | A: sysfs | B: runtime install | C: wrapper image | D: sidecar | E: host |
|---|---|---|---|---|---|
| Zero install delay | ✅ | ❌ | ✅ (after build) | ✅ | ✅ |
| Works offline | ✅ | ❌ | ✅ | ✅ | ✅ |
| Pinned-image purity | ✅ (untouched) | ❌ | ⚠️ (Intel-only build) | ✅ | ✅ |
| Temperature coverage | ✅ (recent kernels) | ✅ (verify) | ✅ (verify) | ✅ (verify) | ✅ (verify) |
| Power coverage | ✅ (recent kernels) | ✅ (verify) | ✅ (verify) | ✅ (verify) | ✅ (verify) |
| Complexity | Low | Medium | Low (one-time build) | High | Low |
| Diverges from "no Dockerfile"? | No | No | Yes (Intel opt-in) | No | No |

---

## 5. Recommendation — staged plan

### Tier 1: sysfs sampling (implement now, zero dependencies)

**No Docker changes. No new tools. Just code.**

1. Extend `telemetry.py` `TelemetrySampler._sample_intel()`:
   - **Phase 1: xe sysfs** (preferred source for freq/temp/power)
     - Discover the xe card from the selected render node (via `/sys/class/drm/cardN/device/renderD*` symlinks)
     - Glob frequency: `/sys/class/drm/card*/device/{cur,min,max}_freq_mhz` and per-GT variants
     - Glob hwmon: find `hwmon` whose `name` matches `xe*` and whose `device` symlink matches the selected card's PCI path
     - Read `temp1_input` (millidegrees C) → convert to °C
     - Read `power1_input` / `power1_average` (milliwatts) → convert to W
   - **Phase 2: xpu-smi fallback** (extended metric list)
     - Request additional metrics: `temperature,power,gpu_frequency` alongside the existing `gpu_utilization,mem_used`
     - Tolerate "not supported" for any metric that returns null — best-effort, never fatal
   - **Phase 3: intel_gpu_top** (i915 only, unchanged)
2. Extend `aggregate()`: add `temp_peak_c`, `temp_avg_c`, `freq_avg_mhz`, `freq_peak_mhz`
3. Add `telemetry_source` to `environment.json`: `"xe-sysfs" | "xpu-smi" | "intel_gpu_top" | "none"`
4. Update `report.py`:
   - Add `("temp_peak_c", "Temp °C")` and `("freq_avg_mhz", "Freq MHz")` to `MODEL_TELEM_LABELS`
   - Add `"telemetry_source"` to `ENV_FIELDS` display
5. Update the README "Known limitations" — remove "Power best-effort" for Intel (or mark it as resolved)

**Risk:** low — pure file reads; degrades gracefully; no new runtime dependency.

### Tier 2: opt-in wrapper image (recommended if Tier 1 lacks coverage on the target kernel)

If the target B70 host runs a kernel < xe-hwmon (older than ~6.15), Tier 1 sysfs won't expose temperature or power. In that case:

- Add `--build-xpu-tools` to `bench.sh` (default: off)
- On first Intel run with this flag, build a wrapper image (one-off):
  ```dockerfile
  FROM vllm/vllm-openai-xpu:0.28.0
  RUN apt-get update && apt-get install -y intel-xpu-smi intel-gpu-tools
  ```
- The built image's ID is recorded in `environment.json` (already supported)
- Subsequent runs reuse the built image (no rebuild)
- Also adds `intel-gpu-tools` (for the `intel_gpu_top` i915 fallback, if needed)

**Alternative:** use Option B (runtime `apt-get`) if a full wrapper build is unwanted — documented as a manual escape hatch.

### Tier 3: optional CPU system metrics

For system-level energy analysis (energy per 1000 output tokens):

- Bind-mount CPU RAPL: `-v /sys/devices/virtual/powercap:/sys/devices/virtual/powercap:ro` (bind over the Docker mask)
- Sample CPU package power from the RAPL interface
- Compute `energy_joules = power_avg_w × duration_s` and `J_per_1k_tokens` in the report
- Note: CPU thermal (`/sys/class/thermal/thermal_zone*`) is already visible — optional add-on

### Rejected approaches

- **D (sidecar):** complexity > benefit. The repo's design principle is a single container with a thin orchestrator.
- **E (host-side):** the container already sees host sysfs; no advantage; violates the container-runtime requirement.
- **B (runtime install alone):** fragile, network-dependent, changes the effective container per run. Only as a documented manual fallback.

---

## 6. Report schema changes (additive, backward-compatible)

### Per-sample (`telemetry_<model>_<config>_<C>.json`)

New fields (all float or null):
```json
{
  "mem_used_gb": 25.3,
  "util_pct": 97.2,
  "power_w": 245.0,
  "temp_c": 72.5,       ← new
  "freq_mhz": 1850.0    ← new
}
```

### Per-aggregate (`telemetry_json` object)

New fields:
```json
{
  "n_samples": 312,
  "duration_s": 310.5,
  "mem_peak_gb": 26.1, "mem_avg_gb": 25.3,
  "util_avg_pct": 96.0, "util_peak_pct": 100.0,
  "power_avg_w": 248.3, "power_peak_w": 310.0,
  "temp_avg_c": 71.2,   ← new
  "temp_peak_c": 82.5,  ← new
  "freq_avg_mhz": 1820.0,  ← new
  "freq_peak_mhz": 1950.0  ← new
}
```

### `report.md` columns

| Model | Config | C | Status | ... | **Temp °C** | **Freq MHz** | Mem peak GB | Util % | Power W |
|---|---|---|---|---|---|---|---|---|---|

(report.py `MODEL_TELEM_LABELS` gains `"temp_peak_c"` and `"freq_avg_mhz"`)

### `environment.json` additions

```json
{
  "telemetry_source": "xe-sysfs",
  "telemetry_metrics": {
    "power": "xe-sysfs",
    "temperature": "xe-sysfs",
    "frequency": "xe-sysfs",
    "utilization": "xpu-smi",
    "memory": "xpu-smi"
  }
}
```

### Optional throttle note

If `temp_peak >= 90°C` and `freq_avg < 0.9 × freq_max` → append `"possible thermal throttle"` to the row's `reason` field (or a new `notes` field). This signals the bottleneck to the reader.

---

## 7. Verification spike (commands for the B70 target box)

These commands must be run on the actual Intel B70 machine to pin down exact paths and available metrics before implementing:

### Host-side (before container)

```bash
# 1. Kernel and driver
uname -r
modinfo xe | grep ^version || echo "xe module not loaded"

# 2. xe sysfs frequency — discover actual path layout
ls /sys/class/drm/card*/device/*freq* 2>/dev/null
ls /sys/class/drm/card*/device/gt*/ *freq* 2>/dev/null
ls /sys/class/drm/card*/device/gt/gt*/*freq* 2>/dev/null

# 3. xe hwmon — discover temperature and power
for h in /sys/class/hwmon/hwmon*; do
  echo "=== $h ==="
  cat "$h/name"
  ls "$h"
done

# 4. Card ↔ render node mapping
ls -la /sys/class/drm/card*/device/renderD*

# 5. CPU RAPL (optional, system-level)
ls /sys/class/powercap/intel-rapl* 2>/dev/null || echo "RAPL not found"
```

### Inside the vLLM XPU container

```bash
# 6. Container sees /sys? (verify /sys is not filtered in Docker default)
docker run --rm --name gpu-bench-telem -it \
  -v /dev/dri:/dev/dri \
  $(for n in /dev/dri/renderD* /dev/dri/card*; do [[ -c "$n" ]] && echo "--device $n"; done) \
  $(getent group video >/dev/null 2>&1 && echo "--group-add video") \
  $(getent group render >/dev/null 2>&1 && echo "--group-add render") \
  vllm/vllm-openai-xpu:0.28.0 \
  sh -c 'echo "=== sysfs ==="; ls /sys/class/drm/card*/device/*freq*; echo "=== hwmon ==="; for h in /sys/class/hwmon/hwmon*; do cat $h/name; ls $h; done; echo "=== tools ==="; command -v xpu-smi || echo "no xpu-smi"; command -v zeinfo || echo "no zeinfo"; command -v intel_gpu_top || echo "no intel_gpu_top"'

# 7. xpu-smi metric enumeration (if present)
docker run --rm --name gpu-bench-telem2 -it \
  -v /dev/dri:/dev/dri \
  $(for n in /dev/dri/renderD* /dev/dri/card*; do [[ -c "$n" ]] && echo "--device $n"; done) \
  $(getent group video >/dev/null 2>&1 && echo "--group-add video") \
  $(getent group render >/dev/null 2>&1 && echo "--group-add render") \
  vllm/vllm-openai-xpu:0.28.0 \
  sh -c 'if command -v xpu-smi >/dev/null 2>&1; then xpu-smi dump -d 0 -m gpu_utilization,mem_used,temperature,power,gpu_frequency; else echo "xpu-smi not present"; fi'

# 8. Correlation check — run --quick and verify telemetry tracks load
./bench.sh --quick --dry-run          # see the exact docker command
# then run with the same flags, monitoring the telemetry file live:
tail -f results/<latest>/telemetry_M1_baseline_1.json
# Expected: freq ramps with load, temp rises over the sweep, power correlates with util
```

### Spike deliverables

After running the above, record:
- Exact kernel version and which sysfs paths exist
- Which xpu-smi metrics return data vs. "not supported"
- The exact `hwmon/name` string for xe
- Whether `intel_gpu_top` sees anything (should be empty on xe)
- A sample during `--quick` showing the data is dynamic (not stale)

---

## 8. Code sketch (Tier 1 — `telemetry.py` changes)

### 8.1 Sysfs discovery helper

```python
def _find_xe_card_for_render(render_path: str) -> str | None:
    """Given a render node path (e.g. /dev/dri/renderD128), find the
    matching /sys/class/drm/cardN device. Returns the card path.

    Mapping: /sys/class/drm/cardN/device contains a 'renderD*' entry
    for every render node belonging to that card."""
    render_name = os.path.basename(render_path)
    import glob
    for card_path in glob.glob("/sys/class/drm/card*"):
        dev = card_path + "/device"
        if os.path.exists(f"{dev}/{render_name}"):
            return dev
    return None
```

### 8.2 Sysfs readers

```python
def _read_sysfs_intel(self, card_dev: str) -> dict:
    """Read frequency, temperature, and power from xe sysfs.
    Returns dict with keys: freq_mhz, temp_c, power_w (float or None)."""
    result = {"freq_mhz": None, "temp_c": None, "power_w": None}

    # Frequency: try multiple glob patterns (single GT, per-GT, multi-GT)
    for pattern in [
        f"{card_dev}/cur_freq_mhz",
        f"{card_dev}/gt0/cur_freq_mhz",
        f"{card_dev}/gt/gt0/cur_freq_mhz",
    ]:
        val = self._read_int(pattern)
        if val is not None:
            result["freq_mhz"] = float(val) / 1000.0  # kHz → MHz
            break
    # For multi-GT: also read gt1 if it exists (media GT)
    # (take the max across GTs as the "active" frequency)

    # Temperature: xe hwmon
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = Path(h, "name").read_text().strip()
        except OSError:
            continue
        if "xe" not in name.lower():
            continue
        # Optional: verify device symlink matches the selected card
        try:
            dev_link = os.readlink(f"{h}/device")
            if card_dev and f"/{os.path.basename(card_dev)}" not in dev_link:
                continue  # different GPU
        except OSError:
            pass
        val = self._read_int(f"{h}/temp1_input")
        if val is not None:
            result["temp_c"] = round(val / 1000.0, 1)  # m°C → °C
        pval = self._read_int(f"{h}/power1_input")
        if pval is not None:
            result["power_w"] = round(pval / 1000.0, 1)  # mW → W
        pval_avg = self._read_int(f"{h}/power1_average")
        if pval_avg is not None:
            result["power_w"] = round(pval_avg / 1000.0, 1)
        break  # found the xe hwmon device

    return result
```

### 8.3 Extended `_sample_intel()`

```python
def _sample_intel(self) -> Optional[dict]:
    idx = str(self.gpu_index if self.gpu_index is not None else 0)
    env = {k: v for k, v in os.environ.items() if k not in ("ONEAPI_DEVICE_SELECTOR",)}
    result = {"mem_used_gb": None, "util_pct": None, "power_w": None,
              "temp_c": None, "freq_mhz": None}

    # Phase 1: xe sysfs (frequency, temperature, power — zero deps)
    # Discover the card device from the selected render node
    # (the container sees all render nodes via --device /dev/dri/*)
    card_dev = _find_xe_card_for_render(f"/dev/dri/renderD{self._current_render}")
    sysfs = self._read_sysfs_intel(card_dev) if card_dev else {}
    result["freq_mhz"] = sysfs.get("freq_mhz")
    result["temp_c"] = sysfs.get("temp_c")
    result["power_w"] = sysfs.get("power_w")

    # Phase 2: xpu-smi (utilization, memory, + extended metrics as fallback)
    # Request additional metrics: temperature, power, frequency
    out = _run(["xpu-smi", "dump", "-d", idx,
                "-m", "gpu_utilization,mem_used,temperature,power,gpu_frequency"],
               env=env, timeout=10.0)
    if out:
        # Parse util
        m = re.search(r"gpu_utilization[\"':\s]+(\d+(?:\.\d+)?)", out)
        if m:
            result["util_pct"] = float(m.group(1))
        # Parse mem (existing heuristic)
        m_mem = re.search(r"mem_used[\"':\s]+(\d+(?:\.\d+)?)", out)
        if m_mem:
            v = float(m_mem.group(1))
            result["mem_used_gb"] = round(v / 1024**3, 2) if v >= 10**8 else round(v / 1024.0, 2)
        # Parse temperature (fallback if sysfs was null)
        if result["temp_c"] is None:
            m_temp = re.search(r"temperature[\"':\s]+(\d+(?:\.\d+)?)", out)
            if m_temp:
                result["temp_c"] = float(m_temp.group(1))
        # Parse power (fallback if sysfs was null)
        if result["power_w"] is None:
            m_power = re.search(r"power[\"':\s]+(\d+(?:\.\d+)?)", out)
            if m_power:
                result["power_w"] = float(m_power.group(1))
        # Parse frequency (fallback if sysfs was null)
        if result["freq_mhz"] is None:
            m_freq = re.search(r"gpu_frequency[\"':\s]+(\d+(?:\.\d+)?)", out)
            if m_freq:
                result["freq_mhz"] = float(m_freq.group(1))

    # Phase 3: intel_gpu_top (i915 only — last resort)
    if all(v is None for v in [result["util_pct"], result["power_w"], result["temp_c"]]):
        out = _run(["intel_gpu_top", "-l", "-s", "1"], env=env, timeout=10.0)
        if out:
            m = re.search(r"GPU Util:\s*(\d+(?:\.\d+)?)%", out)
            if m:
                result["util_pct"] = float(m.group(1))

    # Return if we got something meaningful
    if any(v is not None for v in [result["mem_used_gb"], result["util_pct"],
                                    result["power_w"], result["temp_c"],
                                    result["freq_mhz"]]):
        return result
    return None
```

### 8.4 Extended `aggregate()`

```python
def aggregate(self, samples: list[dict]) -> Optional[dict]:
    # ...existing mem/util/power code...
    temp = [s["temp_c"] for s in samples if s.get("temp_c") is not None]
    freq = [s["freq_mhz"] for s in samples if s.get("freq_mhz") is not None]
    return {
        "n_samples": len(samples),
        "duration_s": round(samples[-1]["t"] - samples[0]["t"], 2),
        "mem_peak_gb": ...,
        "mem_avg_gb": ...,
        "util_avg_pct": ..., "util_peak_pct": ...,
        "power_avg_w": ..., "power_peak_w": ...,
        # New:
        "temp_avg_c": round(sum(temp) / len(temp), 1) if temp else None,
        "temp_peak_c": round(max(temp), 1) if temp else None,
        "freq_avg_mhz": round(sum(freq) / len(freq), 0) if freq else None,
        "freq_peak_mhz": round(max(freq), 0) if freq else None,
    }
```

---

## 9. Summary of recommended actions

| Action | When | Effort | Risk |
|---|---|---|---|
| **Tier 1: sysfs sampler** (code only) | **done** (report.md format unchanged by request — new fields are additive in `telemetry_*.json` / `report.json` telemetry objects + `environment.json.telemetry_metrics`) | Medium (3 files: telemetry.py, run_matrix.py, README) | Low |
| **Spike on B70** (run the §7 commands) | Before Tier 1 implementation | 30 min | None |
| **Tier 2: wrapper image build** (`--build-xpu-tools`) | If Tier 1 lacks temp/power on the host kernel | Low (one Dockerfile + bench.sh flag) | Low |
| **Tier 3: CPU RAPL** (optional system energy) | Post-launch polish | Low | Low |
| Reject: sidecar (D), host-side (E), runtime install (B alone) | — | — | — |

The spike (§7) should run first on the B70 box to confirm:
1. Which sysfs paths exist (frequency layout, hwmon presence)
2. Whether `xpu-smi` is in the vLLM XPU image
3. Which xpu-smi metrics return data vs. "not supported"
4. Whether the data is dynamic (tracks load)

After the spike, implement Tier 1. If temperature/power are missing on the target kernel, propose Tier 2.

---

*Draft — awaiting target-box spike results before implementation.*
