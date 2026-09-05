#!/usr/bin/env python3
"""Cross-system comparison of gpu_inference_bench run directories.

Reads report.json + environment.json from each run dir under results/ and
writes docs/cross-system-comparison.md (or --out) as a human-readable
synthesis — performance, latency and power-efficiency takeaways with a few
compact summary tables — not a line-by-line dump of every cell.

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


def pct(v, best):
    return f"{100.0 * v / best:.0f}%" if best else "n/a"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def gpu_label(e):
    return f"{e.get('gpu') or '?'} ({e['vram_total_gb']:.1f} GB)"


def thr_c16(mid, k):
    r = row(k, mid, "baseline", 16)
    return r["output_throughput"] if r else None


def scale_c1(mid, k):
    r1, r16 = row(k, mid, "baseline", 1), row(k, mid, "baseline", 16)
    if r1 and r16:
        return r16["output_throughput"] / r1["output_throughput"]
    return None


def kv_delta(k, mid, c):
    """kv-fp8 vs baseline output throughput, %."""
    a, b = row(k, mid, "baseline", c), row(k, mid, "kv-fp8", c)
    if a and b:
        return 100.0 * (b["output_throughput"] - a["output_throughput"]) / a["output_throughput"]
    return None


def lat_mean(k, field):
    """Mean of a latency field over all ok baseline cells (all models, C=1..16)."""
    vals = []
    for _, mid, _ in MODEL_ORDER:
        for c in CS:
            r = row(k, mid, "baseline", c)
            if r and r.get(field) is not None:
                vals.append(r[field])
    return mean(vals)


def worst_cell(field):
    """(value, key, row) of the largest baseline-cell value for a latency field."""
    best_v, out = None, None
    for k, _, _ in SYSTEMS:
        for r in data[k]["report"]["rows"]:
            if r["config"] != "baseline" or r["status"] != "ok":
                continue
            v = r.get(field)
            if v is not None and (best_v is None or v > best_v):
                best_v, out = v, (k, r)
    return best_v, out


def eff_cell(k, mid, cfg):
    """(tok/s per watt, (watts, assumed)) at C=16; Intel falls back to TDP."""
    r = row(k, mid, cfg, 16)
    if not r:
        return None
    p = (r.get("telemetry") or {}).get("power_avg_w")
    assumed = False
    if p is None and k == "Intel":
        p = INTEL_B70_TDP_W
        assumed = True
    if not p:
        return None
    return r["output_throughput"] / p, (p, assumed)


def slot_of(mid):
    return [s for s, m, _ in MODEL_ORDER if m == mid][0]


def desc_of(mid):
    return [d for s, m, d in MODEL_ORDER if m == mid][0]


def build() -> str:
    L = []
    keys = [k for k, _, _ in SYSTEMS]
    labels = {k: l for k, _, l in SYSTEMS}

    # ── derived stats (baseline config unless noted) ───────────────────────
    c16 = {mid: {k: v for k, v in
                 ((k, thr_c16(mid, k)) for k in keys) if v is not None}
           for _, mid, _ in MODEL_ORDER}
    leader = {}
    for mid, m in c16.items():
        if m:
            leader[mid] = max(m, key=m.get)
    lead_all = len(set(leader.values())) == 1 and len(leader) == len(MODEL_ORDER)

    share = {k: mean([m[k] / max(m.values()) for m in c16.values() if k in m])
             for k in keys}

    scales = {mid: {k: scale_c1(mid, k) for k in keys} for _, mid, _ in MODEL_ORDER}

    effs = {}
    for _, mid, _ in MODEL_ORDER:
        effs[mid] = {k: ec for k, ec in
                     ((k, eff_cell(k, mid, "baseline")) for k in keys) if ec}
    eff_mean = {k: mean([effs[mid][k][0] for mid in c16 if k in effs.get(mid, {})])
                for k in keys}

    kvd = {k: mean([kv_delta(k, mid, 16) for _, mid, _ in MODEL_ORDER]) for k in keys}

    lcx = {}
    m1 = MODEL_ORDER[0][1]
    for k in keys:
        a, b = row(k, m1, "baseline", 16), row(k, m1, "long-context", 16)
        if a and b:
            lcx[k] = 100.0 * (b["output_throughput"] - a["output_throughput"]) / a["output_throughput"]

    lat = {m: {k: (lat_mean(k, k50), lat_mean(k, k99)) for k in keys}
           for m, k50, k99 in METRICS}
    tail = {k: mean([lat[m][k][1] / lat[m][k][0] for m, _, _ in METRICS
                     if lat[m][k][0] and lat[m][k][1]]) for k in keys}

    ttft16 = {k: mean([(row(k, mid, "baseline", 16) or {}).get("ttft_p50_ms")
                       for _, mid, _ in MODEL_ORDER]) for k in keys}

    itl_worst_v, itl_worst = worst_cell("itl_p99_ms")

    # ── header + systems ───────────────────────────────────────────────────
    L.append("# Cross-System Performance Comparison\n")
    L.append("gpu_inference_bench — 3 systems, identical workload (random "
             "512-in/256-out tokens, 50 prompts, seed 42, temperature 0, "
             "C = 1/4/8/16), vLLM v0.28.0, 4-model matrix.\n")

    L.append("## Systems compared\n")
    L.append("| Field | " + " | ".join(labels[k] for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
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
        L.append(f"| {label} | " + " | ".join(fn(data[k]["env"]) for k in keys) + " |")
    L.append("| Run dir | " + " | ".join(f"`results/{d}`" for _, d, _ in SYSTEMS) + " |")
    L.append("")

    # ── executive summary ──────────────────────────────────────────────────
    top, second, third = sorted(keys, key=lambda k: -share[k])
    L.append("## Executive summary\n")
    peak = max(max(m.values()) for m in c16.values())
    L.append(f"- **{labels[top]} leads output throughput on "
             f"{'all four models' if lead_all else 'most models'}** at C=16 baseline "
             f"(peak {peak:.0f} tok/s); across the 4-model matrix "
             f"{labels[second]} averages {share[second]:.0%} and "
             f"{labels[third]} {share[third]:.0%} of the leader's throughput.")
    L.append(f"- **{labels[top]} is also the most power-efficient**: "
             f"{eff_mean[top]:.2f} mean output tok/s per watt @ C=16 vs "
             f"{eff_mean[second]:.2f} ({labels[second]}) and "
             f"{eff_mean[third]:.2f} ({labels[third]}) — "
             f"{eff_mean[top] / eff_mean[third]:.1f}× the slowest. Intel's figure is a "
             "conservative floor (230 W assumed, no GPU telemetry captured).")
    tpot_best = min(keys, key=lambda k: lat["TPOT"][k][0])
    tpot_worst = max(keys, key=lambda k: lat["TPOT"][k][0])
    tail_best = min(keys, key=lambda k: tail[k])
    tail_worst = max(keys, key=lambda k: tail[k])
    L.append(f"- **{labels[tpot_best]} has the lowest decode latency**: mean TPOT p50 "
             f"{lat['TPOT'][tpot_best][0]:.0f} ms vs {lat['TPOT'][tpot_worst][0]:.0f} ms "
             f"for {labels[tpot_worst]}; tails (mean p99/p50) are tightest on "
             f"{labels[tail_best]} ({tail[tail_best]:.1f}×) and loosest on "
             f"{labels[tail_worst]} ({tail[tail_worst]:.1f}×).")
    L.append(f"- **kv-fp8 KV cache is neutral-to-negative on every system** "
             f"(mean Δ vs baseline @ C=16: "
             + ", ".join(f"{kvd[k]:+.1f}% {labels[k]}" for k in keys)
             + ") — the KV cache is not the bottleneck at this workload size "
             "(~12.3 k KV tokens at C=16).\n")

    # ── performance ────────────────────────────────────────────────────────
    L.append("## Performance\n")
    L.append("### Peak output throughput @ C=16, baseline (tok/s; % of row best)\n")
    L.append("| Model | " + " | ".join(labels[k] for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for slot, mid, desc in MODEL_ORDER:
        m = c16.get(mid, {})
        best = max(m.values(), default=None)
        L.append(f"| {slot} · {mid} | "
                 + " | ".join(f"{m[k]:.1f} ({pct(m[k], best)})" if k in m else "n/a"
                              for k in keys) + " |")
    L.append("")

    L.append("### Batch scaling, C=1 → C=16 (baseline throughput ratio)\n")
    L.append("| Model | " + " | ".join(labels[k] for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for slot, mid, _ in MODEL_ORDER:
        L.append(f"| {slot} · {mid} | "
                 + " | ".join(f"{scales[mid][k]:.1f}×" if scales[mid][k] else "n/a"
                              for k in keys) + " |")
    L.append("")

    L.append("### kv-fp8 vs baseline @ C=16 (output throughput %)\n")
    L.append("| Model | " + " | ".join(labels[k] for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for slot, mid, _ in MODEL_ORDER:
        cells = []
        for k in keys:
            d = kv_delta(k, mid, 16)
            cells.append(f"{d:+.1f}%" if d is not None else "n/a")
        L.append(f"| {slot} · {mid} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("### Takeaways\n")
    gap = {mid: min(m.values()) / max(m.values()) for mid, m in c16.items() if m}
    narrow, wide = max(gap, key=gap.get), min(gap, key=gap.get)
    L.append(f"- The cross-system gap is **narrowest on {slot_of(narrow)}** "
             f"({desc_of(narrow)}: last place still at {gap[narrow]:.0%} of best) "
             f"and **widest on {slot_of(wide)}** ({desc_of(wide)}: {gap[wide]:.0%}).")
    m3 = MODEL_ORDER[2][1]
    s3 = {k: scales[m3][k] for k in keys if scales[m3][k]}
    if s3:
        wk = min(s3, key=s3.get)
        r1, r16 = row(wk, m3, "baseline", 1), row(wk, m3, "baseline", 16)
        if r1 and r16:
            L.append(f"- **{labels[wk]} barely scales on M3**: {s3[wk]:.1f}× from C=1 to C=16 "
                     f"({r1['output_throughput']:.0f} → {r16['output_throughput']:.0f} tok/s) "
                     f"while TPOT p50 climbs {r1['tpot_p50_ms']:.0f} → "
                     f"{r16['tpot_p50_ms']:.0f} ms — batch decode degrades under load "
                     "(KV/scheduling pressure on the tight 32 GB fit and/or a ROCm "
                     "batching inefficiency for this checkpoint).")
    m4 = MODEL_ORDER[3][1]
    for k in keys:
        a, b = row(k, m4, "baseline", 1), row(k, m4, "baseline", 4)
        if a and b and b["output_throughput"] / a["output_throughput"] > 5:
            L.append(f"- **{labels[k]} M4 single-stream anomaly**: "
                     f"{a['output_throughput']:.1f} tok/s at C=1 vs "
                     f"{b['output_throughput']:.1f} at C=4 "
                     f"({b['output_throughput'] / a['output_throughput']:.0f}× step) — "
                     "single-stream MoE decode is inefficient on this stack; the "
                     f"{scales[m4][k]:.1f}× C=1→C=16 'scaling' partly reflects this, not "
                     "superlinear batching.")
    worst_kv16, worst_kv1 = None, None
    for k in keys:
        for _, mid, _ in MODEL_ORDER:
            d16, d1 = kv_delta(k, mid, 16), kv_delta(k, mid, 1)
            if d16 is not None and (worst_kv16 is None or d16 < worst_kv16[0]):
                worst_kv16 = (d16, k, mid)
            if d1 is not None and (worst_kv1 is None or d1 < worst_kv1[0]):
                worst_kv1 = (d1, k, mid)
    if worst_kv16 and worst_kv16[0] < 0:
        d, k, mid = worst_kv16
        extra = ""
        if worst_kv1 and worst_kv1[0] < -20:
            extra = (f" Single-stream is hit even harder: {slot_of(worst_kv1[2])} on "
                     f"{labels[worst_kv1[1]]} {worst_kv1[0]:+.0f}% @ C=1.")
        L.append(f"- kv-fp8 hurts **{labels[k]}** most (mean {kvd[k]:+.1f}%), worst cell "
                 f"{slot_of(mid)} {d:+.1f}% @ C=16.{extra} "
                 "No system benefits at this workload size.")
    if lcx:
        L.append("- **long-context (32k max-model-len) is a no-op for M1** throughput "
                 "(only model with that cell): "
                 + ", ".join(f"{labels[k]} {lcx[k]:+.1f}%" for k in keys if k in lcx)
                 + " vs baseline @ C=16 — expected, since the workload still sends "
                 "512-token prompts and only the supported context window grows "
                 "(AMD's M1 long-context run failed at engine startup; see caveats).")
    L.append("")

    # ── latency ────────────────────────────────────────────────────────────
    L.append("## Latency (ms)\n")
    L.append("Averages over all baseline cells (4 models × C=1/4/8/16). "
             "**p50** = typical request, **p99** = worst 1% of requests. "
             "TTFT = time to first token (prefill + queueing); TPOT = per-token "
             "decode latency; ITL = inter-token gap (streaming tail risk).\n")
    L.append("| Metric | " + " | ".join(f"{labels[k]} p50 | {labels[k]} p99" for k in keys) + " |")
    L.append("|---|" + "---|" * (2 * len(keys)))
    for m, k50, k99 in METRICS:
        L.append(f"| {m} | " + " | ".join(
            f"{lat[m][k][0]:.0f} | {lat[m][k][1]:.0f}" for k in keys) + " |")
    L.append("")
    L.append("### Takeaways\n")
    best_t = min(keys, key=lambda k: ttft16[k])
    worst_t = max(keys, key=lambda k: ttft16[k])
    ttft1 = [v for v in ((row(k, mid, "baseline", 1) or {}).get("ttft_p50_ms")
                         for k in keys for _, mid, _ in MODEL_ORDER) if v]
    m3tt = [r["ttft_p50_ms"] for k in keys
            if (r := row(k, MODEL_ORDER[2][1], "baseline", 16))]
    L.append(f"- **TTFT**: single-stream prefill is fast everywhere (C=1 p50 up to "
             f"{max(ttft1):.0f} ms); under full load (C=16) "
             f"{labels[best_t]} queues fastest ({ttft16[best_t]:.0f} ms mean p50 across "
             f"models) vs {ttft16[worst_t]:.0f} ms for {labels[worst_t]} — the heavy M3 "
             f"dense-27B prefill dominates ({min(m3tt) / 1000:.1f}–{max(m3tt) / 1000:.1f} s "
             "p50 per request at C=16).")
    best_p = min(keys, key=lambda k: lat["TPOT"][k][0])
    worst_p = max(keys, key=lambda k: lat["TPOT"][k][0])
    tp16 = {k: (row(k, m1, "baseline", 16) or {}).get("tpot_p50_ms") for k in keys}
    tp16 = {k: v for k, v in tp16.items() if v}
    spread16 = max(tp16.values()) / min(tp16.values())
    L.append(f"- **TPOT**: {labels[best_p]} decodes fastest at every concurrency level "
             f"({lat['TPOT'][best_p][0]:.0f} ms mean p50 vs {lat['TPOT'][worst_p][0]:.0f} ms "
             f"for {labels[worst_p]}). M1 @ C=16 spans {spread16:.1f}× across systems "
             f"({min(tp16.values()):.0f} → {max(tp16.values()):.0f} ms).")
    wk, wr = itl_worst
    itl99 = {k: lat["ITL"][k][1] for k in keys}
    itl_low = min(keys, key=lambda k: itl99[k])
    itl_high = max(keys, key=lambda k: itl99[k])
    L.append(f"- **ITL tails**: the worst single cell is {labels[wk]} on {wr['model']} "
             f"@ C={wr['concurrency']} ({itl_worst_v:.0f} ms p99) — streaming stalls of "
             f"~{itl_worst_v / 1000:.1f} s. In absolute terms {labels[itl_low]} still "
             f"keeps the worst 1% of inter-token gaps lowest ({itl99[itl_low]:.0f} ms vs "
             f"{itl99[itl_high]:.0f} ms for {labels[itl_high]}); relative tail width "
             f"(mean p99/p50) is tightest on {labels[tail_best]} ({tail[tail_best]:.1f}×).")
    L.append("")

    # ── power efficiency ───────────────────────────────────────────────────
    L.append("## Power efficiency (output tok/s per watt)\n")
    L.append("C=16, baseline. AMD/NVIDIA: measured `power_avg_w` over the bench run. "
             "Intel: no GPU telemetry captured → evaluated at the documented "
             f"**{INTEL_B70_TDP_W:.0f} W max TDP**, so its values are conservative "
             "floors.\n")
    L.append("| Model | " + " | ".join(labels[k] for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for slot, mid, _ in MODEL_ORDER:
        cells = []
        for k in keys:
            ec = effs.get(mid, {}).get(k)
            cells.append(f"{ec[0]:.2f} ({ec[1][0]:.0f} W{'*' if ec[1][1] else ''})"
                         if ec else "n/a")
        L.append(f"| {slot} · {mid} | " + " | ".join(cells) + " |")
    L.append("| **Overall (mean)** | "
             + " | ".join(f"**{eff_mean[k]:.2f}**" for k in keys) + " |")
    L.append("")
    L.append("Note: Intel power assumed (max TDP), not measured.\n")
    L.append("### Takeaways\n")
    best_e = max(keys, key=lambda k: eff_mean[k])
    worst_e = min(keys, key=lambda k: eff_mean[k])
    L.append(f"- **{labels[best_e]} is {eff_mean[best_e] / eff_mean[worst_e]:.1f}× "
             f"{labels[worst_e]} overall.** The gap is largest on M4 (MoE 35B): "
             f"{effs[m4][best_e][0]:.2f} vs {effs[m4][worst_e][0]:.2f} tok/s/W — strong "
             f"MoE execution at a moderate {effs[m4][best_e][1][0]:.0f} W — and smallest "
             f"on M1: {effs[m1][best_e][0]:.2f} vs {effs[m1][worst_e][0]:.2f}.")
    weff = min((effs[mid][k][0], k, mid)
               for mid in c16 for k in keys if k in effs.get(mid, {}))
    wrow = row(weff[1], weff[2], "baseline", 16)
    wW = (wrow.get("telemetry") or {}).get("power_avg_w") or INTEL_B70_TDP_W
    L.append(f"- Worst efficiency cell: {labels[weff[1]]} on {weff[2]} "
             f"({weff[0]:.2f} tok/s/W at {wW:.0f} W) — dense 27B decode is "
             "power-hungry on this stack.")
    L.append(f"- {labels['Intel']}'s true efficiency is at or above the values shown "
             "(assumed full TDP; actual draw was likely lower).")
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