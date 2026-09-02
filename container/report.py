#!/usr/bin/env python3
"""report.py — build report.json + report.md from a run directory.

Reads:
  <out>/cells.json        manifest  {"workload":...,"common_server":...,"models":...,"cells":[...]}
                          (a bare list of cells is also accepted for backward compat)
  <out>/environment.json  environment metadata (optional)
  <out>/bench_*.json      raw vllm bench serve outputs
  <out>/telemetry_*.json  1 Hz GPU sample aggregates

Maps the **actual** vLLM v0.28.0 bench output keys to the report schema.
To adjust mappings, edit BENCH_KEY_MAP below.

Usage:
    python report.py --cells results/<run>/cells.json --out results/<run>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    "duration_s",
]
MODEL_LABELS = [
    "Config", "C", "Status",
    "Req/s", "Tok/s",
    "TTFT p50", "TTFT p99",
    "TPOT p50", "TPOT p99",
    "ITL p50", "ITL p99",
    "Dur s",
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


# ── report builder ─────────────────────────────────────────────────────────
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
                for v in BENCH_KEY_MAP.values():
                    row[v] = None
                rows.append(row)
                continue
            row["status"] = level.get("status", "unknown")
            row["reason"] = level.get("reason")
            bench_path = out_dir / level.get("bench_json", "")
            telem_path = (
                out_dir / level.get("telemetry_json", "")
                if level.get("telemetry_json") else None)
            raw: dict = {}
            if bench_path and bench_path.exists():
                raw = json.loads(bench_path.read_text())
            for rk, nk in BENCH_KEY_MAP.items():
                row[nk] = raw.get(rk)
            telem: dict | None = None
            if telem_path and telem_path.exists():
                telem = json.loads(telem_path.read_text())
            if telem:
                row["telemetry"] = telem
            rows.append(row)
        # Cell never produced per-level data (startup failure / download
        # failure / dry-run) → one row carrying the cell status + reason.
        if not cell.get("concurrency_results"):
            row = {"model": model, "config": config, "concurrency": None,
                   "status": cell_status, "reason": cell_reason}
            for v in BENCH_KEY_MAP.values():
                row[v] = None
            rows.append(row)

    gpu = next((c.get("gpu") for c in cells if c.get("gpu")), None)
    return {
        "schema_version": "1.0",
        "run_id": run_id or out_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": None,
        "workload": workload,
        "models": models,
        "metadata": {
            "gpu": gpu,
            "total_cells": len(cells),
            "total_rows": len(rows),
            "bench_rows": sum(1 for r in rows if r["status"] == "ok"),
            "skip_rows":  sum(1 for r in rows if r["status"].startswith("skipped")),
            "fail_rows":  sum(1 for r in rows if r["status"] == "failed"),
        },
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
    c1 = [r for r in rows if r["concurrency"] == 1 and r["status"] == "ok"]
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
            vals = [_status_str(r) if c == "status" else _fmt(r.get(c))
                    for c in MODEL_COLS]
            tvals = [_fmt(t.get(f)) for f, _ in MODEL_TELEM_LABELS]
            L.append("| " + " | ".join(vals + tvals) + " |")
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
        description="Build report from cells.json + bench results")
    p.add_argument("--cells", required=True, help="Path to cells.json")
    p.add_argument("--out", default=".", help="Output directory")
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
