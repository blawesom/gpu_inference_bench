#!/usr/bin/env python3
"""report.py — build report.json + report.md from cells.json + bench result JSONs.

Maps the **actual** vLLM v0.28.0 bench output keys to the report schema.
To adjust mappings, edit BENCH_KEY_MAP below.

Usage:
    python report.py --cells results/cells.json --out results
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── key mapping: raw v0.28 bench keys → normalized report keys ─────────────
# Source: actual v0.28.0 bench result (pasted by user).
# Add/remove entries here as the schema evolves.
BENCH_KEY_MAP: dict[str, str] = {
    "completed":      "completed",
    "failed":         "failed",
    "request_throughput": "request_throughput",
    "output_throughput":  "output_throughput",
    "duration":       "duration_s",
    # TTFT percentiles
    "p50_ttft_ms":    "ttft_p50_ms",
    "p90_ttft_ms":    "ttft_p90_ms",
    "p99_ttft_ms":    "ttft_p99_ms",
    # TPOT percentiles
    "p50_tpot_ms":    "tpot_p50_ms",
    "p90_tpot_ms":    "tpot_p90_ms",
    "p99_tpot_ms":    "tpot_p99_ms",
    # ITL percentiles
    "p50_itl_ms":     "itl_p50_ms",
    "p90_itl_ms":     "itl_p90_ms",
    "p99_itl_ms":     "itl_p99_ms",
}

# Report columns (in order) for Markdown tables.
REPORT_COLUMNS = [
    "model", "config", "concurrency", "status",
    "completed", "failed",
    "request_throughput", "output_throughput",
    "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
    "itl_p50_ms", "itl_p90_ms", "itl_p99_ms",
    "duration_s",
]

# Markdown column labels (indexed by REPORT_COLUMNS position).
COL_LABELS = [
    "Model", "Config", "C", "Status", "Done", "Fail",
    "Req/s", "Tok/s",
    "TTFT p50", "TTFT p90", "TTFT p99",
    "TPOT p50", "TPOT p90", "TPOT p99",
    "ITL p50", "ITL p90", "ITL p99",
    "Dur s",
]


# ── helper ─────────────────────────────────────────────────────────────────
def _fmt(val: Any) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return str(int(val)) if val == int(val) else f"{val:.2f}"
    return str(val)


# ── report builder ─────────────────────────────────────────────────────────
def _build_report(cells: list[dict], out_dir: Path) -> dict:
    """Build report dict from cells and bench results.

    Each cell (model × config) has ``concurrency_levels`` dict keyed by C,
    each with its own bench_json / telemetry_json. One report row per
    ``(cell, concurrency)`` pair.
    """
    rows: list[dict[str, Any]] = []
    for cell in cells:
        model = cell.get("model_id", "")
        config = cell.get("config_name", "")
        cell_status = cell.get("status", "unknown")
        cell_reason = cell.get("reason")
        for c_str, level in sorted(
                cell.get("concurrency_results", {}).items(),
                key=lambda kv: int(kv[0])):
            row: dict[str, Any] = {
                "model": model, "config": config,
                "concurrency": int(c_str),
            }
            # Cell-level failure/skip: no per-level bench data.
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

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "vllm_version": "v0.28.0",
            "total_cells": len(cells),
            "total_rows": len(rows),
            "bench_rows": sum(1 for r in rows if r["status"] == "ok"),
            "skip_rows":  sum(1 for r in rows if r["status"].startswith("skipped")),
            "fail_rows":  sum(1 for r in rows if r["status"] == "failed"),
        },
        "rows": rows,
    }


# ── writers ────────────────────────────────────────────────────────────────
def _write_json(report: dict, out_dir: Path) -> None:
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def _write_md(report: dict, out_dir: Path) -> None:
    meta = report["metadata"]
    rows = report["rows"]
    L = []
    L.append("# GPU Inference Bench Report\n")
    L.append(f"**vLLM {meta['vllm_version']}** · {meta['generated_at']}\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    for k, label in [
        ("total_cells", "Total cells"),
        ("bench_rows", "Bench rows"),
        ("skip_rows", "Skipped"),
        ("fail_rows", "Failed"),
    ]:
        L.append(f"| {label} | {meta[k]} |")
    L.append("")

    # ── Model summary (C=1, ok only) ─────────────────────────────────
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

    # ── Full table ───────────────────────────────────────────────────
    L.append("## Full Results\n")
    L.append("| " + " | ".join(COL_LABELS) + " |")
    L.append("|" + "|".join(["---"] * len(REPORT_COLUMNS)) + "|")
    for r in rows:
        vals = [_fmt(r.get(c)) for c in REPORT_COLUMNS]
        L.append("| " + " | ".join(vals) + " |")
    L.append("")
    (out_dir / "report.md").write_text("\n".join(L))


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
    cells = json.loads(cells_path.read_text())
    report = _build_report(cells, out_dir)
    _write_json(report, out_dir)
    _write_md(report, out_dir)
    print(f"[report] {len(report['rows'])} rows "
          f"→ {out_dir / 'report.json'}, {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
