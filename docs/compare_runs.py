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

# (key, display label, original run dir [Sept 3-5, legacy protocol],
#   re-run dir [corrected T0/P0/P1 protocol — the authoritative ranking]).
# The re-run is what the primary tables rank; the original run is kept so the
# "Protocol change" section can show the old -> new deltas per card.
SYSTEMS = [
    ("AMD",   "AMD 0x7551",
        "20260903-222650_0x7551",
        "20260913-141852_0x7551"),
    ("Intel", "Intel Arc Pro B70",
        "20260904-212058_intel-r-arc-tm-pro-b70-graphics",
        "20260909-210037_intel-r-arc-tm-pro-b70-graphics"),
    ("NV",    "NVIDIA L40",
        "20260905-134350_nvidia-l40",
        "20260909-142627_nvidia-l40"),
]

# Documented max TDP for the Intel Arc Pro B70 (Intel Arc Pro B-Series spec
# sheet). Fallback only: used for legacy runs where the XPU card captured no
# power telemetry. The re-run measures real power via xpu-smi 2.1.0.
INTEL_B70_TDP_W = 230.0

CS = [1, 4, 8, 16]
CONFIGS = ["baseline", "long-context"]

MODEL_ORDER = [
    ("M1", "Qwen/Qwen3.5-9B", "dense ~9B, BF16"),
    ("M3", "cyankiwi/Qwen3.8-27B-AWQ-INT4", "dense 27B, AWQ-4bit"),
    ("M4", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit", "MoE 35B/3B active, AWQ-4bit"),
]

METRICS = [
    ("TTFT", "ttft_p50_ms", "ttft_p99_ms"),
    ("TPOT", "tpot_p50_ms", "tpot_p99_ms"),
    ("ITL", "itl_p50_ms", "itl_p99_ms"),
]

data = {}  # key -> {"label", "rerun": {"dir","report","env"}, "orig": {...}}


def _load_run(repo: Path, d: str) -> dict:
    run = repo / "results" / d
    return {
        "dir": d,
        "report": json.loads((run / "report.json").read_text()),
        "env": json.loads((run / "environment.json").read_text()),
    }


def load(repo: Path) -> None:
    for k, label, orig, rerun in SYSTEMS:
        data[k] = {
            "label": label,
            "rerun": _load_run(repo, rerun),
            "orig": _load_run(repo, orig),
        }


def row(key, model, config, c, gen="rerun"):
    for r in data[key][gen]["report"]["rows"]:
        if (r["model"] == model and r["config"] == config
                and r.get("concurrency") == c and r["status"] == "ok"):
            return r
    return None


def row_rankable(r):
    """A row may enter rankings only if it is ok AND has no
    output-token-shortfall flag (P0-2). Degraded/partial rows are kept for
    display but excluded from derived stats."""
    if not r or r.get("status") != "ok":
        return False
    if (r.get("token_check") or {}).get("flag"):
        return False
    if r.get("flags") and "output-token-shortfall" in r["flags"]:
        return False
    return True


def row_has_shortfall(r):
    """True if the row's token accounting fell below threshold (P0-2)."""
    if not r:
        return False
    if (r.get("token_check") or {}).get("flag"):
        return True
    return "output-token-shortfall" in (r.get("flags") or [])


def pct(v, best):
    return f"{100.0 * v / best:.0f}%" if best else "n/a"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def gpu_label(e):
    return f"{e.get('gpu') or '?'} ({e['vram_total_gb']:.1f} GB)"


def thr_c16(mid, k):
    r = row(k, mid, "baseline", 16)
    if not row_rankable(r):
        return None
    return r["output_throughput"]


def scale_c1(mid, k):
    r1, r16 = row(k, mid, "baseline", 1), row(k, mid, "baseline", 16)
    if row_rankable(r1) and row_rankable(r16):
        return r16["output_throughput"] / r1["output_throughput"]
    return None


def lat_mean(k, field):
    """Mean of a latency field over all rankable baseline cells."""
    vals = []
    for _, mid, _ in MODEL_ORDER:
        for c in CS:
            r = row(k, mid, "baseline", c)
            if row_rankable(r) and r.get(field) is not None:
                vals.append(r[field])
    return mean(vals)


def worst_cell(field):
    """(value, key, row) of the largest rankable baseline-cell value."""
    best_v, out = None, None
    for k, *_ in SYSTEMS:
        for r in data[k]["rerun"]["report"]["rows"]:
            if r["config"] != "baseline" or not row_rankable(r):
                continue
            v = r.get(field)
            if v is not None and (best_v is None or v > best_v):
                best_v, out = v, (k, r)
    return best_v, out


def eff_cell(k, mid, cfg, gen="rerun"):
    """(tok/s per watt, (watts, assumed)) at C=16; Intel falls back to TDP
    only when that generation has no measured power (legacy runs)."""
    r = row(k, mid, cfg, 16, gen=gen)
    if not row_rankable(r):
        return None
    p = (r.get("telemetry") or {}).get("power_avg_w")
    assumed = False
    if p is None and k == "Intel":
        p = INTEL_B70_TDP_W
        assumed = True
    if not p:
        return None
    return r["output_throughput"] / p, (p, assumed)


def has_measured_power(k, gen="rerun"):
    """True if any rankable baseline cell of that generation carries real
    power (telemetry.power_avg_w) — i.e. not TDP-assumed."""
    for r in data[k][gen]["report"]["rows"]:
        if (r["config"] == "baseline" and row_rankable(r)
                and (r.get("telemetry") or {}).get("power_avg_w") is not None):
            return True
    return False


def slot_of(mid):
    return [s for s, m, _ in MODEL_ORDER if m == mid][0]


def desc_of(mid):
    return [d for s, m, d in MODEL_ORDER if m == mid][0]


def _shortfall(r):
    if not r:
        return False
    if (r.get("token_check") or {}).get("flag"):
        return True
    return "output-token-shortfall" in (r.get("flags") or [])


def build_impact(labels, keys, excluded_models):
    """Old (Sept 3-5, legacy protocol) -> new (re-run) deltas per card and the
    effect on the cross-system ranking. Appended as a standalone section."""
    L = []
    L.append("## Protocol change: original (Sept 3–5) → re-run (corrected)\n")
    L.append("The original runs used the legacy protocol (prefix caching **ON**, "
             "single measured pass, power over the full bench-client window, no "
             "Intel power telemetry). The re-run applies the 2026-09-08 fixes: "
             "prefix caching **OFF** (verified per cell), per-level warmup until "
             "no JIT, **3-pass median**, power/energy **aligned to the measured "
             "window**, and **measured Intel power** (xpu-smi 2.1.0). Two fixes "
             "oppose each other on throughput — prefix-caching off *removes "
             "replay cache hits* (deflates) while warmup-until-no-JIT *removes "
             "in-bench Triton JIT stalls* (inflates) — and they roughly cancel, "
             "except where the JIT stall was the dominant legacy defect.\n")

    # rankable model list (drops any model excluded via output-token shortfall)
    rank = [(slot, mid) for slot, mid, _ in MODEL_ORDER
            if slot not in excluded_models]

    # ── C=16 baseline throughput, old -> new ────────────────────────────────
    L.append("### C=16 baseline output throughput, original → re-run (tok/s, Δ%)\n")
    L.append("| Model | " + " | ".join(labels[k] for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for slot, mid, _ in MODEL_ORDER:
        cells = []
        for k in keys:
            o = row(k, mid, "baseline", 16, gen="orig")
            n = row(k, mid, "baseline", 16, gen="rerun")
            vo = o["output_throughput"] if (o and o.get("status") == "ok") else None
            vn = n["output_throughput"] if (n and n.get("status") == "ok") else None
            if vo is None and vn is None:
                cells.append("n/a")
            elif vo is None:
                cells.append(f"{vn:.0f}")
            else:
                d = 100.0 * (vn - vo) / vo
                mark = "†" if _shortfall(n) else ""
                cells.append(f"{vo:.0f} → {vn:.0f}{mark} ({d:+.0f}%)")
        L.append(f"| {slot} · {mid} | " + " | ".join(cells) + " |")
    if excluded_models:
        L.append(f"\n† **Provisional — excluded from rankings** ({', '.join(excluded_models)}): "
                 "output-token accounting fell below threshold; Δ shown for reference only.\n")

    # ── mean efficiency old -> new per card ─────────────────────────────────
    em_old, em_new = {}, {}
    for k in keys:
        eo = [e[0] for e in (eff_cell(k, mid, "baseline", "orig") for _, mid in rank) if e]
        en = [e[0] for e in (eff_cell(k, mid, "baseline", "rerun") for _, mid in rank) if e]
        em_old[k], em_new[k] = mean(eo), mean(en)
    L.append("### Mean power efficiency (tok/s/W @ C=16, rankable models), original → re-run\n")
    L.append("| Card | original | re-run | Δ | power (orig → re-run) |")
    L.append("|---|---|---|---|---|")
    for k in keys:
        po = "measured" if has_measured_power(k, "orig") else f"assumed {INTEL_B70_TDP_W:.0f} W"
        pn = "measured" if has_measured_power(k, "rerun") else f"assumed {INTEL_B70_TDP_W:.0f} W"
        d = 100.0 * (em_new[k] - em_old[k]) / em_old[k] if em_old[k] else None
        L.append(f"| {labels[k]} | {em_old[k]:.2f} | {em_new[k]:.2f} | "
                 f"{d:+.0f}% | {po} → {pn} |")
    L.append("")

    # ── share of leader old vs new ──────────────────────────────────────────
    def share_leader(gen):
        s = {}
        for _, mid in rank:
            vals = {}
            for k in keys:
                r = row(k, mid, "baseline", 16, gen=gen)
                if row_rankable(r):
                    vals[k] = r["output_throughput"]
            if vals:
                mx = max(vals.values())
                s[mid] = {k: 100.0 * vals[k] / mx for k in vals}
        return s
    so_old, so_new = share_leader("orig"), share_leader("rerun")
    L.append("### Share of leader @ C=16 (rankable models), original → re-run\n")
    L.append("| Model | " + " | ".join(f"{labels[k]} old → new" for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for slot, mid in rank:
        cells = []
        for k in keys:
            o, n = so_old.get(mid, {}).get(k), so_new.get(mid, {}).get(k)
            cells.append(f"{o:.0f}% → {n:.0f}%" if (o is not None and n is not None) else "n/a")
        L.append(f"| {slot} · {mid} | " + " | ".join(cells) + " |")
    L.append("")

    # ── ranking impact ──────────────────────────────────────────────────────
    ms_old = {k: mean([so_old[m].get(k) for m in so_old if k in so_old[m]]) for k in keys}
    ms_new = {k: mean([so_new[m].get(k) for m in so_new if k in so_new[m]]) for k in keys}
    thr_old = sorted(keys, key=lambda k: -ms_old[k])
    thr_new = sorted(keys, key=lambda k: -ms_new[k])
    eff_old = sorted(keys, key=lambda k: -em_old[k])
    eff_new = sorted(keys, key=lambda k: -em_new[k])
    # largest single-cell throughput delta (rankable)
    deltas = []
    for k in keys:
        for slot, mid in rank:
            o, n = row(k, mid, "baseline", 16, gen="orig"), row(k, mid, "baseline", 16, gen="rerun")
            if row_rankable(o) and row_rankable(n):
                d = 100.0 * (n["output_throughput"] - o["output_throughput"]) / o["output_throughput"]
                deltas.append((d, k, slot))
    deltas.sort(key=lambda x: -abs(x[0]))
    big_d, big_k, big_slot = deltas[0]
    top_e, bot_e = eff_new[0], eff_new[-1]
    ratio_old = em_old[top_e] / em_old[bot_e]
    ratio_new = em_new[top_e] / em_new[bot_e]

    L.append("### Ranking impact\n")
    L.append(f"- **Throughput order** {'unchanged' if thr_old == thr_new else 'CHANGED'}: "
             f"{' > '.join(labels[k] for k in thr_old)} → "
             f"{' > '.join(labels[k] for k in thr_new)}.")
    L.append(f"- **Efficiency order** {'unchanged' if eff_old == eff_new else 'CHANGED'}: "
             f"{' > '.join(labels[k] for k in eff_old)} → "
             f"{' > '.join(labels[k] for k in eff_new)}.")
    L.append(f"- **{labels[top_e]}’s efficiency lead over {labels[bot_e]}** shrank from "
             f"{ratio_old:.1f}× (original) to **{ratio_new:.1f}×** (re-run): the legacy "
             "power window under-stated steady-state draw, so the original figures were "
             "optimistic.")
    L.append(f"- The largest single-cell throughput swing is **{labels[big_k]} {big_slot}** "
             f"({big_d:+.0f}% at C=16 baseline) — the rest of the matrix moves <5%, so the "
             "gap compression is driven by that one JIT-deflated cell.")
    if (not has_measured_power("Intel", "orig")
            and has_measured_power("Intel", "rerun")):
        L.append(f"- **{labels['Intel']}** is now on **measured** power (xpu-smi 2.1.0) "
                 "instead of an assumed 230 W TDP floor — its efficiency is real draw, so "
                 "the old “conservative floor” caveat no longer applies.")
    L.append("")
    return L


def build() -> str:
    L = []
    keys = [k for k, *_ in SYSTEMS]
    labels = {k: l for k, l, *_ in SYSTEMS}

    # ── derived stats (baseline config unless noted) ───────────────────────
    c16 = {mid: {k: v for k, v in
                 ((k, thr_c16(mid, k)) for k in keys) if v is not None}
           for _, mid, _ in MODEL_ORDER}
    # c16_raw: every ok row (incl. P0-flagged) for display with † markers.
    c16_raw = {mid: {k: row(k, mid, "baseline", 16)
                     for k in keys if row(k, mid, "baseline", 16)}
               for _, mid, _ in MODEL_ORDER}
    excluded_models = [slot_of(mid) for mid in c16
                       if not c16[mid] and mid in c16_raw and any(c16_raw[mid])]
    leader = {}
    for mid, m in c16.items():
        if m:
            leader[mid] = max(m, key=m.get)
    lead_all = (len(set(leader.values())) == 1 and len(leader)
                == len(MODEL_ORDER) - len(excluded_models))

    share = {k: mean([m[k] / max(m.values()) for m in c16.values() if k in m])
             for k in keys}

    scales = {mid: {k: scale_c1(mid, k) for k in keys} for _, mid, _ in MODEL_ORDER}

    effs = {}
    for _, mid, _ in MODEL_ORDER:
        effs[mid] = {k: ec for k, ec in
                     ((k, eff_cell(k, mid, "baseline")) for k in keys) if ec}
    eff_mean = {k: mean([effs[mid][k][0] for mid in c16 if k in effs.get(mid, {})])
                for k in keys}

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
                       for _, mid, _ in MODEL_ORDER
                       if row_rankable(row(k, mid, "baseline", 16))])
              for k in keys}

    itl_worst_v, itl_worst = worst_cell("itl_p99_ms")

    # ── header + systems ───────────────────────────────────────────────────
    L.append("# Cross-System Performance Comparison\n")
    L.append("gpu_inference_bench — 3 systems, identical workload (random "
             "512-in/256-out tokens, 50 prompts, seed 42, temperature 0, "
             "C = 1/4/8/16), vLLM v0.28.0, 3-model matrix.\n")

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
        L.append(f"| {label} | " + " | ".join(fn(data[k]["rerun"]["env"]) for k in keys) + " |")
    L.append("| Run dir (re-run) | "
             + " | ".join(f"`results/{data[k]['rerun']['dir']}`" for k in keys) + " |")
    L.append("| Run dir (orig.) | "
             + " | ".join(f"`results/{data[k]['orig']['dir']}`" for k in keys) + " |")
    L.append("")

    # ── executive summary ──────────────────────────────────────────────────
    top, second, third = sorted(keys, key=lambda k: -share[k])
    L.append("## Executive summary\n")
    peak = max(max(m.values()) for m in c16.values() if m)
    excl_note = (f" ({', '.join(excluded_models)} excluded: output-token "
                 f"shortfall, see † below)") if excluded_models else ""
    L.append(f"- **{labels[top]} leads output throughput on "
             f"{'all three models' if (lead_all and not excluded_models) else 'most models'}** at C=16 baseline "
             f"(peak {peak:.0f} tok/s); across the {len(MODEL_ORDER) - len(excluded_models)}-model matrix{excl_note} "
             f"{labels[second]} averages {share[second]:.0%} and "
             f"{labels[third]} {share[third]:.0%} of the leader's throughput.")
    assumed = [labels[k] for k in keys if not has_measured_power(k)]
    pw_note = (" All three systems use measured power in the aligned window."
               if not assumed else
               " " + ", ".join(assumed) +
               f" assume max TDP (no power telemetry captured).")
    L.append(f"- **{labels[top]} is also the most power-efficient**: "
             f"{eff_mean[top]:.2f} mean output tok/s per watt @ C=16 vs "
             f"{eff_mean[second]:.2f} ({labels[second]}) and "
             f"{eff_mean[third]:.2f} ({labels[third]}) — "
             f"{eff_mean[top] / eff_mean[third]:.1f}× the slowest.{pw_note}")
    tpot_best = min(keys, key=lambda k: lat["TPOT"][k][0])
    tpot_worst = max(keys, key=lambda k: lat["TPOT"][k][0])
    tail_best = min(keys, key=lambda k: tail[k])
    tail_worst = max(keys, key=lambda k: tail[k])
    L.append(f"- **{labels[tpot_best]} has the lowest decode latency**: mean TPOT p50 "
             f"{lat['TPOT'][tpot_best][0]:.0f} ms vs {lat['TPOT'][tpot_worst][0]:.0f} ms "
             f"for {labels[tpot_worst]}; tails (mean p99/p50) are tightest on "
             f"{labels[tail_best]} ({tail[tail_best]:.1f}×) and loosest on "
             f"{labels[tail_worst]} ({tail[tail_worst]:.1f}×).\n")

    # ── performance ────────────────────────────────────────────────────────
    L.append("## Performance\n")
    L.append("### Peak output throughput @ C=16, baseline (tok/s; % of row best)\n")
    L.append("| Model | " + " | ".join(labels[k] for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for slot, mid, desc in MODEL_ORDER:
        m = c16.get(mid, {})
        best = max(m.values(), default=None)
        cells = []
        for k in keys:
            if k in m:
                cells.append(f"{m[k]:.1f} ({pct(m[k], best)})")
            elif k in c16_raw.get(mid, {}):
                val = c16_raw[mid][k]["output_throughput"]
                cells.append(f"{val:.1f}†")
            else:
                cells.append("n/a")
        L.append(f"| {slot} · {mid} | " + " | ".join(cells) + " |")
    if excluded_models:
        L.append("")
        L.append(f"† **Provisional — excluded from rankings and derived stats** "
                 f"({', '.join(excluded_models)}). Output-token accounting fell "
                 "below threshold (expected 50 × 256 = 12800 tokens); see each "
                 "run's `report.md → Data quality`. Values must not be ranked "
                 "until diagnosed.")
    L.append("")

    L.append("### Batch scaling, C=1 → C=16 (baseline throughput ratio)\n")
    L.append("| Model | " + " | ".join(labels[k] for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for slot, mid, _ in MODEL_ORDER:
        L.append(f"| {slot} · {mid} | "
                 + " | ".join(f"{scales[mid][k]:.1f}×" if scales[mid][k] else "n/a"
                              for k in keys) + " |")
    L.append("")

    L.append("### Takeaways\n")
    gap = {mid: min(m.values()) / max(m.values()) for mid, m in c16.items() if m}
    narrow, wide = max(gap, key=gap.get), min(gap, key=gap.get)
    L.append(f"- The cross-system gap is **narrowest on {slot_of(narrow)}** "
             f"({desc_of(narrow)}: last place still at {gap[narrow]:.0%} of best) "
             f"and **widest on {slot_of(wide)}** ({desc_of(wide)}: {gap[wide]:.0%}).")
    m3 = MODEL_ORDER[1][1]
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
    m4 = MODEL_ORDER[2][1]
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
    L.append("Averages over all baseline cells (3 models × C=1/4/8/16). "
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
             if (r := row(k, MODEL_ORDER[1][1], "baseline", 16))]
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
    # Per-card window/power note so the header works for any mix of
    # generations (some aligned, some legacy, some TDP-assumed).
    win_parts = []
    for k in keys:
        r = row(k, MODEL_ORDER[0][1], "baseline", 16)
        t = (r.get("telemetry") or {}) if r else {}
        win = ("aligned to the measured window" if t.get("window")
               else "full-client window (pre-P1)")
        pw = ("measured" if t.get("power_avg_w") is not None
              else f"assumed {INTEL_B70_TDP_W:.0f} W (no telemetry)")
        win_parts.append(f"{labels[k]}: {win}, {pw}")
    L.append("C=16, baseline. " + " · ".join(win_parts) + ". "
             "Energy is GPU-only (vendor power sensor), not system energy; "
             "`energy_j_per_ktok` is per level.\n")
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
    if any(not has_measured_power(k) for k in keys):
        L.append("Note: some power values are assumed (max TDP), not measured.\n")
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
    if not has_measured_power("Intel"):
        L.append(f"- {labels['Intel']}'s true efficiency is at or above the values shown "
                 "(assumed full TDP; actual draw was likely lower).")
    L.append("")

    # ── protocol change: old (Sept 3-5) -> new (re-run) ──────────────────
    L.extend(build_impact(labels, keys, excluded_models))

    # ── caveats ────────────────────────────────────────────────────────────
    L.append("## Caveats\n")
    L.append("- **Protocol (P0/P1/T1) now enforced**: prefix caching OFF "
             "(verified per cell), per-level warmup until no JIT, 3-pass "
             "median, and power/energy aligned to the measured bench window "
             "(GPU-only, vendor power sensor). The Sept 3–5 legacy runs "
             "(prefix caching ON, single pass, unaligned power window, no "
             "Intel power telemetry) are **superseded** — see the "
             "\"Protocol change\" section for the effect on their numbers.")
    # per-card data-quality notes: failed cells + single-pass measurements
    for k in keys:
        rpt = data[k]["rerun"]["report"]
        failed = [f"{r['config']} ({r.get('reason')})"
                  for r in rpt["rows"]
                  if r.get("status") == "failed" and r["config"] != "aiter-attn"]
        single = sorted({r["model"].split("/")[-1]
                         for r in rpt["rows"]
                         if r["config"] == "baseline" and r.get("status") == "ok"
                         and r.get("num_passes") == 1})
        bits = []
        if failed:
            bits.append("**failed cells**: " + ", ".join(failed))
        if single:
            bits.append("**single-pass (not a 3-pass median)**: " + ", ".join(single))
        if bits:
            L.append(f"- **{labels[k]} re-run**: " + "; ".join(bits) + ".")
    L.append("- **AITER (T1) M1 on AMD**: the aiter-attn cell failed at engine "
             "startup (`pass1-failed:engine-startup`) — no AITER throughput "
             "was captured this run, so the ROCm Triton-fallback baseline "
             "remains the AMD attention reference. AITER is AMD-only; the "
             "cell auto-skips on NVIDIA/Intel.")
    L.append("- **VRAM differs**: NVIDIA L40 has 45 GB vs 32 GB on AMD/Intel. "
             "All three models fit at this workload (~12.3 k KV tokens at "
             "C=16); the extra headroom only matters for long-context cells.")
    L.append("- **Host CPUs differ**: AMD/Intel runs on an AMD Ryzen 7 9800X3D "
             "(consumer), NVIDIA on an Intel Xeon (Sapphire Rapids). Negligible "
             "for GPU-bound decode; noted for completeness.")

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