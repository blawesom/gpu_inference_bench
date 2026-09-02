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
#   MODELS          comma list, e.g. M1,M2      (optional)
#   CONFIGS         comma list, e.g. baseline   (optional)
#   CONCURRENCY     comma list, e.g. 1,8,16     (optional)
#   QUICK           1 → M1 only, baseline+kv-fp8, C=1,8
#   KEEP_WEIGHTS    1 → do not delete weights per model
#   HF_HOME         default /hf-cache
#   HOST_UID / HOST_GID   for chown (skipped when uid 0)
set -uo pipefail

RESULTS="${RESULTS:-/results}"
export HF_HOME="${HF_HOME:-/hf-cache}"

echo "=== entrypoint: vendor=${GPU_VENDOR:-auto} models=${MODELS:-<all>} \
configs=${CONFIGS:-<all>} conc=${CONCURRENCY:-<all>} quick=${QUICK:-0} \
keep_weights=${KEEP_WEIGHTS:-0} ==="

ARGS=(--config /bench/config/models.yaml --results "$RESULTS"
      --vendor "${GPU_VENDOR:-auto}")
[[ -n "${GPU_INDEX:-}" ]]     && ARGS+=(--gpu-index "$GPU_INDEX")
[[ -n "${MODELS:-}" ]]        && ARGS+=(--models "$MODELS")
[[ -n "${CONFIGS:-}" ]]       && ARGS+=(--configs "$CONFIGS")
[[ -n "${CONCURRENCY:-}" ]]   && ARGS+=(--concurrency "$CONCURRENCY")
[[ "${QUICK:-0}" == "1" ]]    && ARGS+=(--quick)
[[ "${KEEP_WEIGHTS:-0}" == "1" ]] && ARGS+=(--keep-weights)

echo "--- run_matrix ---"
python3 /bench/container/run_matrix.py "${ARGS[@]}"
MATRIX_RC=$?

echo "--- report ---"
if [[ -f "$RESULTS/cells.json" ]]; then
    python3 /bench/container/report.py --cells "$RESULTS/cells.json" \
        --out "$RESULTS" || echo "WARN: report.py failed (cells.json present)"
else
    echo "WARN: no cells.json — skipping report"
fi

# Restore ownership for the host user (container runs as root).
if [[ "${HOST_UID:-0}" != "0" ]]; then
    chown -R "${HOST_UID}:${HOST_GID:-${HOST_UID}}" "$RESULTS" 2>/dev/null || true
    chown -R "${HOST_UID}:${HOST_GID:-${HOST_UID}}" "$HF_HOME" 2>/dev/null || true
fi

echo "=== entrypoint done (run_matrix rc=$MATRIX_RC) ==="
exit "$MATRIX_RC"