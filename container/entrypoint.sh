#!/usr/bin/env bash
# entrypoint.sh — in-container bootstrap for the GPU inference benchmark.
#
# Called by bench.sh via:  docker run ... --entrypoint bash $IMAGE /bench/container/entrypoint.sh
#
# GPU selection is done INSIDE the container by run_matrix.py (the host may not
# have a GPU management CLI, and on AMD the host sysfs card number does not map
# to the HIP index). This script just assembles the CLI and runs the matrix +
# report, then restores host ownership of the output dirs.
#
# Environment (set by bench.sh):
#   GPU_VENDOR      amd | nvidia | intel        (required)
#   GPU_INDEX       physical index override     (optional)
#   MODELS          comma list, e.g. M1,M2 or M1-M4   (optional)
#   CONFIGS         comma list, e.g. baseline   (optional)
#   CONCURRENCY     comma list, e.g. 1,8,16     (optional)
#   QUICK           1 → M1 only, baseline+kv-fp8, C=1,8
#   VALIDATE        1 → run the VRAM-fit validator instead of the matrix
#   KEEP_WEIGHTS    deprecated no-op (weights are kept by default now)
#   DELETE_WEIGHTS  1 → delete weights per model (old behavior)
#   HF_HOME         default /hf-cache
#   HOST_UID / HOST_GID   for chown (skipped when uid 0)
set -uo pipefail

usage() {
    cat <<'EOF'
Usage: bash entrypoint.sh [OPTIONS]

In-container bootstrap for the GPU inference benchmark.  Called by bench.sh
via docker run --entrypoint bash.  All configuration is via environment
variables (set by bench.sh), not CLI flags.

ENVIRONMENT (required):
  GPU_VENDOR          amd | nvidia | intel    (required)
  RESULTS             output directory         (default /results)

ENVIRONMENT (optional):
  GPU_INDEX           physical GPU index override
  MODELS              comma list (ranges ok), e.g. M1,M2 or M1-M4  (default: all)
  CONFIGS             comma list, e.g. baseline
  CONCURRENCY         comma list, e.g. 1,8,16
  QUICK               1 → M1 only, baseline+kv-fp8, C=1,8
  VALIDATE            1 → run VRAM-fit validator instead of the matrix
  KEEP_WEIGHTS        deprecated no-op (weights kept by default)
  DELETE_WEIGHTS      1 → delete weights per model (old behavior)
  HF_HOME             HF cache dir              (default /hf-cache)
  HOST_UID/HOST_GID   for chown of output dirs (skipped when uid 0)
  SERVER_START_TIMEOUT  server health-wait budget (seconds)

OPTIONS:
  -h, --help          Show this help and exit

This script assembles the CLI and runs run_matrix.py (or validate_fit.py
when VALIDATE=1), then restores host ownership of the output dirs.
EOF
    exit 0
}

# Handle -h/--help before anything else
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage ;;
    esac
done

set -uo pipefail

RESULTS="${RESULTS:-/results}"
export HF_HOME="${HF_HOME:-/hf-cache}"

echo "=== entrypoint: vendor=${GPU_VENDOR:-auto} models=${MODELS:-<all>} \
configs=${CONFIGS:-<all>} conc=${CONCURRENCY:-<all>} quick=${QUICK:-0} \
validate=${VALIDATE:-0} keep_weights=${KEEP_WEIGHTS:-0} ==="

ARGS=(--config /bench/config/models.yaml --results "$RESULTS"
      --vendor "${GPU_VENDOR:-auto}")
[[ -n "${GPU_INDEX:-}" ]]     && ARGS+=(--gpu-index "$GPU_INDEX")
[[ -n "${MODELS:-}" ]]        && ARGS+=(--models "$MODELS")
[[ -n "${CONFIGS:-}" ]]       && ARGS+=(--configs "$CONFIGS")
[[ -n "${CONCURRENCY:-}" ]]   && ARGS+=(--concurrency "$CONCURRENCY")
[[ "${QUICK:-0}" == "1" ]]    && ARGS+=(--quick)
[[ "${KEEP_WEIGHTS:-0}" == "1" ]] && ARGS+=(--keep-weights)
[[ "${DELETE_WEIGHTS:-0}" == "1" ]] && ARGS+=(--delete-weights)

# Validation is a distinct mode: static estimate + optional live probe, no
# concurrency sweep. validate_fit.py only takes the subset of flags below.
VAL_ARGS=(--config /bench/config/models.yaml --results "$RESULTS"
          --vendor "${GPU_VENDOR:-auto}")
[[ -n "${GPU_INDEX:-}" ]] && VAL_ARGS+=(--gpu-index "$GPU_INDEX")
[[ -n "${MODELS:-}" ]]    && VAL_ARGS+=(--models "$MODELS")

if [[ "${VALIDATE:-0}" == "1" ]]; then
    echo "--- validate_fit ---"
    python3 /bench/container/validate_fit.py "${VAL_ARGS[@]}"
    MATRIX_RC=$?
else
    echo "--- run_matrix ---"
    python3 /bench/container/run_matrix.py "${ARGS[@]}"
    MATRIX_RC=$?
fi

# Locate this run's directory. run_matrix.py writes the run-id basename to
# $RESULTS/.latest (per-run dir: <YYYYMMDD-HHMMSS>_<gpu-slug>/); fall back to
# $RESULTS itself for dry-runs / older layouts.
RUN_ID="$(cat "$RESULTS/.latest" 2>/dev/null || true)"
RUN_DIR="$RESULTS"
if [[ -n "$RUN_ID" && -d "$RESULTS/$RUN_ID" ]]; then
    RUN_DIR="$RESULTS/$RUN_ID"
fi

echo "--- report ---"
if [[ "${VALIDATE:-0}" == "1" ]]; then
    VAL_DIR="$(ls -dt "$RESULTS"/validate_* 2>/dev/null | head -1 || true)"
    if [[ -n "$VAL_DIR" && -f "$VAL_DIR/fit_report.md" ]]; then
        cat "$VAL_DIR/fit_report.md"
    else
        echo "WARN: no fit_report.md found under $RESULTS/validate_*"
    fi
elif [[ -f "$RUN_DIR/cells.json" ]]; then
    python3 /bench/container/report.py --cells "$RUN_DIR/cells.json" \
        --out "$RUN_DIR" || echo "WARN: report.py failed (cells.json present)"
else
    echo "WARN: no cells.json — skipping report"
fi

# Restore ownership for the host user (container runs as root).
if [[ "${HOST_UID:-0}" != "0" ]]; then
    chown -R "${HOST_UID}:${HOST_GID:-${HOST_UID}}" "$RESULTS" 2>/dev/null || true
    chown -R "${HOST_UID}:${HOST_GID:-${HOST_UID}}" "$HF_HOME" 2>/dev/null || true
fi

echo "=== entrypoint done (rc=$MATRIX_RC) → $RUN_DIR ==="
exit "$MATRIX_RC"