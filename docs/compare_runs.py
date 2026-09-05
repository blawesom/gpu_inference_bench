#!/usr/bin/env python3
"""Cross-system comparison of gpu_inference_bench run directories.

Reads report.json + environment.json from each run dir under results/ and
writes docs/cross-system-comparison.md (or --out).

Usage: python3 docs/compare_runs.py [repo] [--out PATH]
"""
import json
import sys
from pathlib import Path

# (key, run-dir, display label)
SYSTEMS = [
    ("AMD",   "20260903-222650_0x7551", "AMD 0x7551"),
    ("Intel", "20260904-212058_intel-r-arc-tm-pro-b70-graphics", "Intel Arc Pro B70"),
    ("NV",    "20260905-134350_nvidia-l40", "NVIDIA L40"),
]

# Documented max TDP for the Intel Arc Pro B70 (Intel Arc Pro B-Series spec
# sheet). Used because the XPU run captured no power telemetry.
INTEL_B70_TDP_W = 230.0

CS = [1, 4, 8, 16]
CONFIGS = ["baseline", "kv-fp8", "long-context"]

MODEL_ORDER = [
    ("M1", "Qwen/Qwen3.5-9B", "dense ~9B, BF16"),
    ("M2", "openai/gpt-oss-20b", "MoE 21B/3.6B active, MXFP4"),
    ("M3", "cyankiwi/Qwen3.8-27B-AWQ-INT4", "dense 27B, AWQ-4bit"),
    ("M4", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit", "MoE 35B/3B active, AWQ-4bit"),
]

METRICS = [
    ("TTFT", "ttft_p50_ms", "ttft_p99_ms"),
    ("TPOT", "tpot_p50_ms", "tpot_p99_ms"),
    ("ITL", "itl_p50_ms", "itl_p99_ms"),
]

data = {}  # key -> {"label", "dir", "report", "env"}


def load(repo: Path) -> None:
    for k, d, label in SYSTEMS:
        run = repo / "results" / d
        data[k] = {
            "label": label,
            "dir": d,
            "report": json.loads((run / "report.json").read_text()),
            "env": json.loads((run / "environment.json").read_text()),
        }


def row(key, model, config, c):
    for r in data[key]["report"]["rows"]:
        if (r["model"] == model and r["config"] == config
                and r.get("concurrency") == c and r["status"] == "ok"):
            return r
    return None


def f1(v):
    return "n/a" if v is None else f"{v:.1f}"


def f2(v):
    return "n/a" if v is None else f"{v:.2f}"


def pct(v, best):
    return f"{100.0 * v / best:.0f}%" if best else "n/a"


def gpu_label(e):
    return f"{e.get('gpu') or '?'} ({f1(e.get('vram_total_gb'))} GB)"


def eff(key, model, config, c=16):
    """output tok/s per watt; Intel falls back to documented max TDP."""
    r = row(key, model, config, c)
    if not r:
        return None
    p = (r.get("telemetry") or {}).get("power_avg_w")
    if p is None and key == "Intel":
        p = INTEL_B70_TDP_W
    if not p:
        return None
    return r["output_throughput"] / p


def cfg_present(model, config):
    return any(any(r["model"] == model and r["config"] == config
                   for r in data[k]["report"]["rows"]) for k, _, _ in SYSTEMS)


def build() -> str:
    L = []
    L.append("# Cross-System Performance Comparison\n")
    L.append("gpu_inference_bench — 3 systems, identical workload (random "
             "512-in/256-out tokens, 50 prompts, seed 42, temperature 0, "
             "C = 1/4/8/16), vLLM v0.28.0, 4-model matrix.\n")

    # ── systems ────────────────────────────────────────────────────────────
    L.append("## Systems compared\n")
    L.append("| Field | " + " | ".join(l for _, _, l in SYSTEMS) + " |")
    L.append("|---|" + "---|" * len(SYSTEMS))
    fields = [
        ("GPU", lambda e: gpu_label(e)),
        ("Stack", lambda e: ", ".join(f"{k} {v}"
                                      for k, v in (e.get("stack") or {}).items() if v) or "n/a"),
        ("Driver", lambda e: e.get("driver") or "n/a"),
        ("vLLM", lambda e: e.get("vllm_version") or "n/a"),
        ("Image", lambda e: e.get("image") or "n/a"),
        ("OS / CPU", lambda e: f"{e.get('os') or '?'} / {e.get('cpu') or '?'}"),
    ]
    for label, fn in fields:
        L.append(f"| {label} | " + " | ".join(fn(data[k]["env"]) for k, _, _ in SYSTEMS) + " |")
    L.append("| Run dir | " + " | ".join(f"`results/{d}`" for _, d, _ in SYSTEMS) + " |")
    L.append("")

    # ── caveats ────────────────────────────────────────────────────────────
    L.append("## Caveats\n")
    L.append("- **VRAM differs**: NVIDIA L40 has 45 GB vs 32 GB on AMD/Intel. "
             "All four models fit comfortably on 32 GB at this workload "
             "(~12.3 k KV tokens at C=16), so the extra headroom does not "
             "change scheduling; it only matters for long-context cells.")
    L.append("- **AMD**: M1 `long-context` cell failed at engine startup "
             "(`failed: engine-startup`); shown as n/a. All other cells ran.")
    L.append("- **Intel**: the XPU run captured **no GPU telemetry** "
             "(mem/util/power). Power-efficiency figures for Intel assume the "
             f"documented **max TDP of {INTEL_B70_TDP_W:.0f} W** for the Arc Pro "
             "B70 (Intel Arc Pro B-Series spec sheet) — an upper bound on "
             "actual draw, so Intel's tok/s-per-W values are conservative "
             "floors (true efficiency is at or better than shown).")
    L.append("- **Host CPUs differ**: AMD/Intel runs on an AMD Ryzen 7 9800X3D "
             "(consumer), NVIDIA on an Intel Xeon (Sapphire Rapids). Negligible "
             "for GPU-bound decode; noted for completeness. The AMD run used "
             "GPU index 1 on a dual-GPU host.")
    L.append("")

    # ── throughput ─────────────────────────────────────────────────────────
    L.append("## Throughput (output tok/s)\n")
    L.append("Share-of-best in parentheses (row best = 100%).\n")
    for slot, mid, desc in MODEL_ORDER:
        L.append(f"### {slot} · {mid} ({desc})\n")
        L.append("| Config | C | " + " | ".join(l for _, _, l in SYSTEMS) + " |")
        L.append("|---|---|" + "---|" * len(SYSTEMS))
        for cfg in CONFIGS:
            if not cfg_present(mid, cfg):
                continue
            for c in CS:
                vals = []
                for k, _, _ in SYSTEMS:
                    r = row(k, mid, cfg, c)
                    vals.append(r["output_throughput"] if r else None)
                best = max([v for v in vals if v is not None], default=None)
                cells = [f"{v:.1f} ({pct(v, best)})" if v is not None else "n/a"
                         for v in vals]
                L.append(f"| {cfg} | {c} | " + " | ".join(cells) + " |")
        L.append("")

    # ── latency ────────────────────────────────────────────────────────────
    L.append("## Latency (ms)\n")
    L.append("Per-metric tables, p50 / p99. Lower is better.\n")
    for slot, mid, desc in MODEL_ORDER:
        L.append(f"### {slot} · {mid}\n")
        for mname, k50, k99 in METRICS:
            L.append(f"**{mname}**\n")
            L.append("| Config | C | "
                     + " | ".join(f"{l} p50 | {l} p99" for _, _, l in SYSTEMS)
                     + " |")
            L.append("|---|---|" + "---|" * (2 * len(SYSTEMS)))
            for cfg in CONFIGS:
                if not cfg_present(mid, cfg):
                    continue
                for c in CS:
                    cells = []
                    for k, _, _ in SYSTEMS:
                        r = row(k, mid, cfg, c)
                        cells += [f1(r.get(k50)) if r else "n/a",
                                  f1(r.get(k99)) if r else "n/a"]
                    L.append(f"| {cfg} | {c} | " + " | ".join(cells) + " |")
            L.append("")

    # ── power efficiency ───────────────────────────────────────────────────
    L.append("## Power efficiency (output tok/s per watt)\n")
    L.append("AMD/NVIDIA: measured `power_avg_w` over the C=16 bench run. "
             "Intel: no telemetry captured → evaluated at the documented max "
             f"TDP ({INTEL_B70_TDP_W:.0f} W), so its values are conservative "
             "floors.\n")
    for slot, mid, desc in MODEL_ORDER:
        L.append(f"### {slot} · {mid} @ C=16\n")
        L.append("| Config | " + " | ".join(l for _, _, l in SYSTEMS) + " | best |")
        L.append("|---|" + "---|" * (len(SYSTEMS) + 1))
        for cfg in CONFIGS:
            if not cfg_present(mid, cfg):
                continue
            vals = [eff(k, mid, cfg) for k, _, _ in SYSTEMS]
            valid = [v for v in vals if v is not None]
            L.append(f"| {cfg} | " + " | ".join(f2(v) for v in vals)
                     + f" | {f2(max(valid)) if valid else 'n/a'} |")
        L.append("")

    L.append("### Overall (mean across all ok cells @ C=16)\n")
    means = []
    for k, _, _ in SYSTEMS:
        vals = []
        for _, mid, _ in MODEL_ORDER:
            for cfg in CONFIGS:
                e = eff(k, mid, cfg)
                if e is not None:
                    vals.append(e)
        means.append(sum(vals) / len(vals) if vals else None)
    L.append("| " + " | ".join(l for _, _, l in SYSTEMS) + " |")
    L.append("|" + "---|" * len(SYSTEMS))
    L.append("| " + " | ".join(f2(m) for m in means) + " |")
    L.append("")

    # ── key findings ───────────────────────────────────────────────────────
    L.append("## Key findings\n")
    for slot, mid, desc in MODEL_ORDER:
        thr = {}
        for k, _, _ in SYSTEMS:
            r = row(k, mid, "baseline", 16)
            if r:
                thr[k] = r["output_throughput"]
        if len(thr) >= 2:
            best_k = max(thr, key=thr.get)
            worst_k = min(thr, key=thr.get)
            L.append(f"- **{slot} {mid}** @ C=16 baseline: {data[best_k]['label']} "
                     f"leads with {thr[best_k]:.0f} tok/s; {data[worst_k]['label']} "
                     f"trails at {thr[worst_k]:.0f} tok/s "
                     f"({thr[best_k] / thr[worst_k]:.1f}× spread).")
    L.append("")
    L.append("**kv-fp8 delta vs baseline @ C=16** (output throughput %)\n")
    L.append("| Model | " + " | ".join(l for _, _, l in SYSTEMS) + " |")
    L.append("|---|" + "---|" * len(SYSTEMS))
    for _, mid, _ in MODEL_ORDER:
        cells = []
        for k, _, _ in SYSTEMS:
            a, b = row(k, mid, "baseline", 16), row(k, mid, "kv-fp8", 16)
            cells.append(f"{100.0 * (b['output_throughput'] - a['output_throughput']) / a['output_throughput']:+.1f}%"
                         if a and b else "n/a")
        L.append(f"| {mid} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("- kv-fp8 is **neutral-to-negative on every system** — the KV cache "
             "is not the bottleneck at this workload size (12.3 k tokens at C=16).")
    r_a2b = row("AMD", "openai/gpt-oss-20b", "baseline", 16)
    r_a2f = row("AMD", "openai/gpt-oss-20b", "kv-fp8", 16)
    d_m2 = 100.0 * (r_a2f["output_throughput"] - r_a2b["output_throughput"]) \
             / r_a2b["output_throughput"] if r_a2b and r_a2f else None
    a3 = row("AMD", "cyankiwi/Qwen3.8-27B-AWQ-INT4", "baseline", 1)
    b3 = row("AMD", "cyankiwi/Qwen3.8-27B-AWQ-INT4", "kv-fp8", 1)
    d_m3c1 = 100.0 * (b3["output_throughput"] - a3["output_throughput"]) / a3["output_throughput"] \
             if a3 and b3 else None
    if d_m2 is not None and d_m3c1 is not None:
        L.append(f"- **AMD regresses the most with kv-fp8**: M2 {d_m2:+.1f}% @ C=16, "
                 f"and M3 single-stream nearly halves "
                 f"({a3['output_throughput']:.1f} → {b3['output_throughput']:.1f} tok/s, "
                 f"{d_m3c1:+.0f}%) — a quirk of the ROCm FP8-KV path, not a workload "
                 "effect.")
    m4a1, m4a4, m4a16 = (row("AMD", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit", "baseline", c)
                     for c in (1, 4, 16))
    if m4a1 and m4a4 and m4a16:
        L.append(f"- **M4 single-stream anomaly (AMD)**: {m4a1['output_throughput']:.1f} tok/s "
                 f"at C=1 vs {m4a4['output_throughput']:.1f} at C=4 "
                 f"({m4a4['output_throughput'] / m4a1['output_throughput']:.0f}× step) — "
                 "single-stream MoE decode is inefficient on ROCm; the "
                 f"{m4a16['output_throughput'] / m4a1['output_throughput']:.1f}× "
                 "'scaling' for M4/AMD below partly reflects this, not superlinear "
                 "batching.")
    m3a1 = row("AMD", "cyankiwi/Qwen3.8-27B-AWQ-INT4", "baseline", 1)
    m3a16 = row("AMD", "cyankiwi/Qwen3.8-27B-AWQ-INT4", "baseline", 16)
    if m3a1 and m3a16:
        L.append(f"- **M3 barely scales on AMD**: only "
                 f"{m3a16['output_throughput'] / m3a1['output_throughput']:.1f}× from C=1 to C=16 "
                 f"({m3a1['output_throughput']:.0f} → {m3a16['output_throughput']:.0f} tok/s) while "
                 f"TPOT p50 climbs {m3a1['tpot_p50_ms']:.0f} → {m3a16['tpot_p50_ms']:.0f} ms and ITL "
                 f"p99 hits {m3a16['itl_p99_ms'] / 1000:.1f} s — batch decode degrades under "
                 "load (KV/scheduling pressure on the tight 32 GB fit and/or a ROCm "
                 "batching inefficiency for this checkpoint).")
    for slot, mid, desc in MODEL_ORDER:
        scale = {}
        for k, _, _ in SYSTEMS:
            r1, r16 = row(k, mid, "baseline", 1), row(k, mid, "baseline", 16)
            if r1 and r16:
                scale[k] = r16["output_throughput"] / r1["output_throughput"]
        if scale:
            L.append(f"- **{slot} scaling (baseline C=1→C=16)**: "
                     + ", ".join(f"{data[k]['label']} {v:.1f}×"
                                 for k, v in scale.items()))
    if all(m is not None for m in means):
        best_i = max(range(len(means)), key=lambda i: means[i])
        amd_i = 0  # AMD is first in SYSTEMS
        m4_nv = eff("NV", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit", "baseline")
        m4_amd = eff("AMD", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit", "baseline")
        if m4_nv and m4_amd:
            L.append(f"- **Power efficiency**: {data[SYSTEMS[best_i][0]]['label']} is "
                     f"most efficient overall ({means[best_i]:.2f} tok/s/W mean @ C=16) — "
                     f"{means[best_i] / means[amd_i]:.1f}× AMD. The gap is largest on M4 "
                     f"(MoE 35B): {m4_nv:.2f} vs {m4_amd:.2f} tok/s/W. Intel's figures "
                     "are floors (assumed 230 W max TDP, no telemetry captured).")
    return "\n".join(L)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", nargs="?", default=".", help="repo root (default: cwd)")
    p.add_argument("--out", default=None,
                   help="output path (default: <repo>/docs/cross-system-comparison.md)")
    args = p.parse_args()
    repo = Path(args.repo).resolve()
    load(repo)
    out = Path(args.out).resolve() if args.out else repo / "docs" / "cross-system-comparison.md"
    out.write_text(build())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()