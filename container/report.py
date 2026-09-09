#!/usr/bin/env python3
"""report.py — build report.json + report.md from a run directory.

Reads:
  <out>/cells.json        manifest  {"workload":...,"common_server":...,"models":...,"cells":[...]}
                          (a bare list of cells is also accepted for backward compat)
  <out>/environment.json  environment metadata (optional)
  <out>/bench_*.json      raw vllm bench serve outputs
  <out>/telemetry_*.json  1 Hz GPU sample aggregates

P0 protocol (2026-09-08 review):
  * Multi-pass cells: per-level `passes[]` are aggregated to the MEDIAN, with
    the pass spread (output_throughput_min/max) kept on the row. Legacy
    single-bench cells are unchanged.
  * Output-token accounting: expected = num_prompts × random_output_len
    (ignore_eos pins the length). Levels below token_check_threshold are
    flagged output-token-shortfall — values are flagged, never corrected.
  * Prefix caching: the effective enable_prefix_caching value is parsed from
    the server log (manifest if present, otherwise the log file) and any
    cell with it effectively ON is called out in the Data quality section.

Maps the **actual** vLLM v0.28.0 bench output keys to the report schema.
To adjust mappings, edit BENCH_KEY_MAP below.

Usage:
    python report.py --cells results/<run>/cells.json --out results/<run>
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# effective prefix-caching value, as printed in the vLLM engine config dump
_PREFIX_CACHE_RE = re.compile(r"enable_prefix_caching=(True|False)")

# ── key mapping: raw v0.28 bench keys → normalized report keys ─────────────
BENCH_KEY_MAP: dict[str, str] = {
    "completed":      "completed",
    "failed":         "failed",
    "request_throughput": "request_throughput",
    "output_throughput":  "output_throughput",
    "duration":       "duration_s",
    "p50_ttft_ms":    "ttft_p50_ms",
    "p90_ttft_ms":    "ttft_p90_ms",
    "p99_ttft_ms":    "ttft_p99_ms",
    "p50_tpot_ms":    "tpot_p50_ms",
    "p90_tpot_ms":    "tpot_p90_ms",
    "p99_tpot_ms":    "tpot_p99_ms",
    "p50_itl_ms":     "itl_p50_ms",
    "p90_itl_ms":     "itl_p90_ms",
    "p99_itl_ms":     "itl_p99_ms",
}

# Columns for the per-model tables.
MODEL_COLS = [
    "config", "concurrency", "status",
    "request_throughput", "output_throughput",
    "ttft_p50_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p99_ms",
    "itl_p50_ms", "itl_p99_ms",
    "duration_s", "flags",
]
MODEL_LABELS = [
    "Config", "C", "Status",
    "Req/s", "Tok/s (median)",
    "TTFT p50", "TTFT p99",
    "TPOT p50", "TPOT p99",
    "ITL p50", "ITL p99",
    "Dur s", "Flags",
]

# Telemetry fields appended to each per-model row.
MODEL_TELEM_LABELS = [("mem_peak_gb", "Mem peak GB"),
                      ("util_avg_pct", "Util %"),
                      ("power_avg_w", "Power W")]

# Metadata table for the report header.
META_COLS = [
    ("total_cells", "Total cells"),
    ("bench_rows",  "Bench rows"),
    ("skip_rows",   "Skipped"),
    ("fail_rows",   "Failed"),
]

# Environment fields to display, in order.
ENV_FIELDS = [
    ("vendor",         "Vendor"),
    ("gpu",            "GPU"),
    ("gpu_index_in_container", "GPU index (in container)"),
    ("vram_total_gb",  "VRAM (GB)"),
    ("driver",         "Driver"),
    ("stack",          "Stack"),
    ("os",             "OS"),
    ("kernel",         "Kernel"),
    ("cpu",            "CPU"),
    ("cpu_cores",      "CPU cores"),
    ("ram_gb",         "RAM (GB)"),
    ("gpu_kernel_modules", "GPU kernel modules"),
    ("docker_version", "Docker"),
    ("image",          "Image"),
    ("image_id",       "Image ID"),
    ("vllm_version",   "vLLM"),
]


# ── helper ─────────────────────────────────────────────────────────────────
def _fmt(val: Any) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return str(int(val)) if val == int(val) else f"{val:.2f}"
    return str(val)


def _status_str(r: dict) -> str:
    s = r.get("status") or "unknown"
    reason = r.get("reason")
    if reason:
        reason = str(reason).replace("\n", " ")[:80]
        return f"{s}: {reason}"
    return s


def _token_check_compute(raw: dict, workload: dict | None) -> dict | None:
    """P0-2 output-token accounting (works on legacy runs too).

    expected = num_prompts × random_output_len (ignore_eos pins the length).
    Below token_check_threshold the check is flagged; the numbers are
    NEVER corrected.
    """
    if not workload:
        return None
    try:
        expected = (int(workload.get("num_prompts", 50))
                    * int(workload.get("random_output_len", 256)))
    except (TypeError, ValueError):
        return None
    actual = raw.get("total_output_tokens")
    if actual is None or expected <= 0:
        return None
    threshold = float(workload.get("token_check_threshold", 0.9))
    ratio = actual / expected
    check = {"expected": expected, "actual": int(actual),
             "ratio": round(ratio, 3)}
    if ratio < threshold:
        check["flag"] = "output-token-shortfall"
    return check


def _effective_prefix_caching(log_text: str | None) -> bool | None:
    """Parse the effective enable_prefix_caching from a vLLM server log."""
    if not log_text:
        return None
    m = _PREFIX_CACHE_RE.search(log_text)
    return m.group(1) == "True" if m else None


def _read_log_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _collect_pass_entries(out_dir: Path, level: dict,
                          workload: dict | None) -> list[dict]:
    """Normalize one concurrency level into per-pass metric entries.

    New layout:  level["passes"] = [{bench_json, telemetry_json, status,
    reason, jit_during_bench, ...}, ...]  (one entry per measured pass).
    Legacy:      level["bench_json"] = single raw bench file.

    Each entry: {"metrics": {normalized keys}, "status", "reason",
                 "token_check" (always recomputed from the raw file),
                 "jit_during_bench", "telemetry"?}.
    """
    entries: list[dict] = []

    def _entry_from_raw(raw: dict, status: str, reason, jit) -> dict:
        entry: dict = {"status": status, "reason": reason,
                       "metrics": ({nk: raw.get(rk) for rk, nk in
                                    BENCH_KEY_MAP.items()}
                                   if status == "ok" else None),
                       "token_check": _token_check_compute(raw, workload),
                       "jit_during_bench": jit}
        return entry

    if level.get("passes"):
        for p in level["passes"]:
            if p.get("status") != "ok":
                entries.append({"metrics": None,
                                "status": p.get("status", "failed"),
                                "reason": p.get("reason"),
                                "token_check": None,
                                "jit_during_bench": p.get("jit_during_bench")})
                continue
            bp = out_dir / p.get("bench_json", "")
            raw = json.loads(bp.read_text()) if bp.exists() else {}
            entry = _entry_from_raw(raw, "ok", None,
                                    p.get("jit_during_bench"))
            tp = (out_dir / p["telemetry_json"]
                  if p.get("telemetry_json") else None)
            if tp and tp.exists():
                try:
                    entry["telemetry"] = json.loads(tp.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            entries.append(entry)
    elif level.get("bench_json"):
        raw = {}
        bp = out_dir / level.get("bench_json", "")
        if bp.exists():
            try:
                raw = json.loads(bp.read_text())
            except json.JSONDecodeError:
                pass
        status = level.get("status", "ok")
        entry = _entry_from_raw(raw, status, level.get("reason"), None)
        tp = (out_dir / level["telemetry_json"]
              if level.get("telemetry_json") else None)
        if tp and tp.exists():
            try:
                entry["telemetry"] = json.loads(tp.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        entries.append(entry)
    return entries


# ── report builder ─────────────────────────────────────────────────────────
def _median(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _row_from_entries(out_dir: Path, level: dict,
                      workload: dict | None) -> dict:
    """Aggregate per-pass entries into one report row (median of metrics)."""
    entries = _collect_pass_entries(out_dir, level, workload)
    ok = [e for e in entries if e.get("metrics") is not None]
    row: dict[str, Any] = {"status": level.get("status", "unknown"),
                           "reason": level.get("reason")}
    for nk in BENCH_KEY_MAP.values():
        row[nk] = _median([e["metrics"][nk] for e in ok])
    if ok:
        ots = [e["metrics"]["output_throughput"] for e in ok
               if e["metrics"].get("output_throughput") is not None]
        if len(ots) > 1:
            row["output_throughput_min"] = min(ots)
            row["output_throughput_max"] = max(ots)
        # token check: prefer a flagged entry, else any check present
        tc = next((e["token_check"] for e in ok
                   if (e.get("token_check") or {}).get("flag")), None)
        if tc is None:
            tc = next((e["token_check"] for e in ok if e.get("token_check")),
                      None)
        if tc:
            row["token_check"] = tc
        # telemetry: from the last ok pass (representative of the regime)
        telem = next((e["telemetry"] for e in reversed(ok)
                      if e.get("telemetry")), None)
        if telem:
            row["telemetry"] = telem
    if entries:
        row["num_passes"] = len(entries)
        row["passes_ok"] = len(ok)
    jit = any(e.get("jit_during_bench") for e in entries)
    if jit:
        row["jit_during_bench"] = True
    flags = []
    if (row.get("token_check") or {}).get("flag"):
        flags.append("output-token-shortfall")
    if row.get("jit_during_bench"):
        flags.append("jit-during-bench")
    if row["status"] == "degraded":
        flags.append("partial-passes")
    row["flags"] = flags
    return row


def _data_quality(cells: list[dict], out_dir: Path,
                  rows: list[dict]) -> list[dict]:
    """P0 audit list: prefix caching, warmup stability, flagged rows.

    The effective prefix-caching value comes from the cell manifest if
    recorded (new runs), otherwise it is parsed from the server log (this
    retro-covers legacy runs, e.g. the 2026-09-03 run where it was ON).
    """
    dq: list[dict] = []
    for cell in cells:
        tag = f"{cell.get('model')} / {cell.get('config')}"
        sf = cell.get("server_flags") or {}
        eff_pc = sf.get("effective_prefix_caching")
        if eff_pc is None:
            # legacy manifest: parse the server log ourselves
            log_name = cell.get("server_log")
            if log_name:
                eff_pc = _effective_prefix_caching(_read_log_text(
                    out_dir / log_name))
            if eff_pc is not None:
                sf["effective_prefix_caching"] = eff_pc  # cache in manifest
        if eff_pc is True:
            dq.append({"cell": tag,
                       "issue": "prefix-caching-ON",
                       "detail": ("server log shows enable_prefix_caching=True "
                                  "— prompt replay across concurrency levels "
                                  "inflates throughput (see 2026-09-08 review)")})
        elif eff_pc is False:
            dq.append({"cell": tag, "issue": "prefix-caching-off",
                       "detail": "verified enable_prefix_caching=False"})
        else:
            dq.append({"cell": tag, "issue": "prefix-caching-unverified",
                       "detail": "could not parse enable_prefix_caching from "
                                  "server log"})
        for c_str, level in sorted(cell.get("concurrency_results", {}).items(),
                                   key=lambda kv: int(kv[0])):
            w = level.get("warmup") or {}
            if w.get("passes") and not w.get("stable"):
                dq.append({"cell": f"{tag} C={c_str}",
                           "issue": "warmup-unstable",
                           "detail": (f"{w['passes']} warmup pass(es) and "
                                      f"{len(w.get('compilations', []))} "
                                      f"JIT compilation(s) still appearing")})
    for r in rows:
        for f in r.get("flags") or []:
            dq.append({"cell": (f"{r.get('model')} / {r.get('config')} "
                                f"C={r.get('concurrency')}"),
                       "issue": f,
                       "detail": (f"output tokens {r['token_check']['actual']}"
                                  f"/{r['token_check']['expected']} "
                                  f"({r['token_check']['ratio']:.0%})"
                                  if f == "output-token-shortfall"
                                  else f"see bench_*.json / server logs")})
    return dq


def _build_report(cells: list[dict], out_dir: Path,
                  workload: dict | None, models: dict | None,
                  run_id: str | None = None) -> dict:
    """Build report dict from cells and bench results."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        model = cell.get("model_id", "")
        config = cell.get("config", "")
        cell_status = cell.get("status", "unknown")
        cell_reason = cell.get("reason")
        for c_str, level in sorted(
                cell.get("concurrency_results", {}).items(),
                key=lambda kv: int(kv[0])):
            row: dict[str, Any] = {
                "model": model, "config": config,
                "concurrency": int(c_str),
            }
            if cell_status != "ok":
                row["status"] = cell_status
                row["reason"] = cell_reason
                row["flags"] = []
                for v in BENCH_KEY_MAP.values():
                    row[v] = None
                rows.append(row)
                continue
            agg = _row_from_entries(out_dir, level, workload)
            row.update(agg)
            rows.append(row)
        # Cell never produced per-level data (startup failure / download
        # failure / dry-run) → one row carrying the cell status + reason.
        if not cell.get("concurrency_results"):
            row = {"model": model, "config": config, "concurrency": None,
                   "status": cell_status, "reason": cell_reason, "flags": []}
            for v in BENCH_KEY_MAP.values():
                row[v] = None
            rows.append(row)

    data_quality = _data_quality(cells, out_dir, rows)
    gpu = next((c.get("gpu") for c in cells if c.get("gpu")), None)
    return {
        "schema_version": "1.1",
        "run_id": run_id or out_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": None,
        "workload": workload,
        "models": models,
        "metadata": {
            "gpu": gpu,
            "total_cells": len(cells),
            "total_rows": len(rows),
            "bench_rows": sum(1 for r in rows
                              if r["status"] in ("ok", "degraded")),
            "skip_rows":  sum(1 for r in rows
                              if r["status"].startswith("skipped")),
            "fail_rows":  sum(1 for r in rows if r["status"] == "failed"),
            "flagged_rows": sum(1 for r in rows if r.get("flags")),
        },
        "data_quality": data_quality,
        "rows": rows,
    }


def _env_display(v: Any) -> str:
    """Format a single environment value for the report."""
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        parts = [f"{k} {val}" for k, val in v.items() if val]
        return ", ".join(parts) if parts else "n/a"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else "n/a"
    return str(v)


def _write_md(report: dict, out_dir: Path) -> None:
    """Render report.md: env block, C=1 summary, per-model detail tables."""
    meta = report["metadata"]
    rows = report["rows"]
    env = report.get("environment")
    workload = report.get("workload")
    models = report.get("models") or {}
    gpu = report.get("metadata", {}).get("gpu")

    L = []
    # ── Header ───────────────────────────────────────────────────────────
    L.append("# GPU Inference Bench Report\n")
    L.append(f"**{gpu or 'GPU'}** · Run: {report['run_id']} · "
             f"vLLM {meta.get('vllm_version') or 'v0.28.0'} · "
             f"{report['generated_at']}\n")

    L.append("| Metric | Value |")
    L.append("|---|---|")
    for k, label in META_COLS:
        L.append(f"| {label} | {meta[k]} |")
    if report.get("workload"):
        w = report["workload"]
        L.append(f"| Input tokens | {w.get('random_input_len', '?')} |")
        L.append(f"| Output tokens | {w.get('random_output_len', '?')} |")
        L.append(f"| Prompts | {w.get('num_prompts', '?')} |")
        L.append(f"| Concurrency levels | {w.get('concurrency_levels', '?')} |")
        if int(w.get("num_passes", 1)) > 1:
            L.append(f"| Measured passes | {w.get('num_passes')} (server restart "
                     f"between passes; metrics are the **median**, spread in "
                     f"report.json `output_throughput_min/max`) |")
            if w.get("warmup_max_passes"):
                L.append(f"| Warmup | per-level, up to {w.get('warmup_max_passes')} "
                         f"extra passes until no new JIT compilations "
                         f"({w.get('warmup_prompts', '?')} prompts each) |")
    L.append("")

    # ── Environment ──────────────────────────────────────────────────────
    if env:
        L.append("## Environment\n")
        L.append("| Field | Value |")
        L.append("|---|---|")
        for key, label in ENV_FIELDS:
            v = env.get(key)
            L.append(f"| {label} | {_env_display(v)} |")
        L.append("")

    # ── Model summary (C=1) ──────────────────────────────────────────────
    c1 = [r for r in rows if r["concurrency"] == 1
          and r["status"] in ("ok", "degraded")]
    if c1:
        L.append("## Model Summary (Concurrency = 1)\n")
        L.append("| Model | Config | Done | Fail | Req/s | Tok/s | TTFT p99 | TPOT p99 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in c1:
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["model"], r["config"],
                _fmt(r["completed"]), _fmt(r["failed"]),
                _fmt(r.get("request_throughput")),
                _fmt(r.get("output_throughput")),
                _fmt(r.get("ttft_p99_ms")),
                _fmt(r.get("tpot_p99_ms")),
            ))
        L.append("")

    # ── Per-model tables ─────────────────────────────────────────────────
    # Build model_id → label from the manifest; fallback to model_id itself.
    model_labels: dict[str, str] = {}
    model_order: list[str] = []
    for mk, md in (models or {}).items():
        if isinstance(md, dict) and md.get("id"):
            model_labels[md["id"]] = f"{mk} · {md['id']}"
            model_order.append(md["id"])

    def model_sort_key(model_id: str) -> tuple:
        idx = model_order.index(model_id) if model_id in model_order else 999
        return (idx, model_id)

    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        mid = r.get("model") or "?"
        if mid not in groups:
            groups[mid] = []
            order.append(mid)
        groups[mid].append(r)
    order.sort(key=model_sort_key)

    for mid in order:
        label = model_labels.get(mid, mid)
        L.append(f"## {label}\n")
        L.append("| " + " | ".join(MODEL_LABELS)
                 + " | " + " | ".join(ml for _, ml in MODEL_TELEM_LABELS)
                 + " |")
        header_sep = "|".join(["---"] * (len(MODEL_LABELS) + len(MODEL_TELEM_LABELS)))
        L.append("|" + header_sep + "|")

        # Sort rows: by config, then concurrency (None sorts first).
        for r in sorted(groups[mid],
                        key=lambda r: (r["config"], r["concurrency"] or 0)):
            t = r.get("telemetry") or {}
            vals = [_status_str(r) if c == "status"
                    else (", ".join(r.get(c)) if c == "flags" and r.get(c)
                          else ("—" if c == "flags" else _fmt(r.get(c))))
                    for c in MODEL_COLS]
            tvals = [_fmt(t.get(f)) for f, _ in MODEL_TELEM_LABELS]
            L.append("| " + " | ".join(vals + tvals) + " |")
        L.append("")

    # ── Data quality (P0 protocol audit) ─────────────────────────────────
    dq = report.get("data_quality") or []
    L.append("## Data quality\n")
    issues = [d for d in dq if d["issue"] != "prefix-caching-off"]
    n_off = sum(1 for d in dq if d["issue"] == "prefix-caching-off")
    if not dq:
        L.append("No data-quality entries recorded for this run (pre-P0 "
                 "manifest?). Re-run report.py after the P0 protocol for a "
                 "full audit.")
    elif not issues:
        L.append("No issues detected.")
    else:
        L.append("| Cell | Issue | Detail |")
        L.append("|---|---|---|")
        for d in issues:
            L.append(f"| {d['cell']} | **{d['issue']}** | {d['detail']} |")
        if n_off:
            L.append("")
            L.append(f"(prefix caching verified OFF in {n_off} cell(s) — "
                     f"not listed above)")
    L.append("")

    # Write
    (out_dir / "report.md").write_text("\n".join(L))


# ── cell loader (accepts both manifest dict and legacy list) ──────────────
def _load_cells(cells_path: Path):
    """Return (cells_list, workload_or_None, models_or_None, run_id_or_None)."""
    doc = json.loads(cells_path.read_text())
    if isinstance(doc, dict):
        return (doc.get("cells", []), doc.get("workload"),
                doc.get("models"), doc.get("run_id"))
    return doc, None, None, None


# ── writers ────────────────────────────────────────────────────────────────
def _write_json(report: dict, out_dir: Path) -> None:
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Build report.json + report.md from cells.json and the raw "
                    "bench/telemetry JSON artifacts of one benchmark run.")
    p.add_argument("--cells", required=True, help="Path to cells.json")
    p.add_argument("--out", default=".", help="Output directory (default: cells.json dir)")
    args = p.parse_args()
    cells_path = Path(args.cells)
    out_dir = Path(args.out)
    if not cells_path.exists():
        print(f"ERROR: {cells_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load cells + manifest context
    cells, workload, models, m_run_id = _load_cells(cells_path)

    # Load environment.json (best-effort; report carries environment: null if absent)
    env_path = out_dir / "environment.json"
    env = None
    if env_path.exists():
        try:
            env = json.loads(env_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    # Build
    report = _build_report(cells, out_dir, workload, models, run_id=m_run_id)
    if env is not None:
        report["environment"] = env
        # Pull vllm_version from environment for the header
        if env.get("vllm_version"):
            report["metadata"]["vllm_version"] = env["vllm_version"]

    _write_json(report, out_dir)
    _write_md(report, out_dir)
    print(f"[report] {len(report['rows'])} rows "
          f"→ {out_dir / 'report.json'}, {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
