#!/usr/bin/env bash
# clean.sh — standalone host script to remove cached model weights from the HF cache.
#
# Weights are kept in the cache by default after each benchmark run so re-runs
# skip the ~20-25 GB re-download.  Use this script to free that disk space.
#
# The HF cache structure is:
#   <cache_dir>/hub/models--<org>--<name>/
#
# Model keys (from config/models.yaml): M1 M2 M3 M4
#   M1 → Qwen/Qwen3.5-9B
#   M2 → openai/gpt-oss-20b
#   M3 → cyankiwi/Qwen3.8-27B-AWQ-INT4
#   M4 → cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
#
# Usage:
#   ./clean.sh                        # remove ALL cached weights (~80 GB)
#   ./clean.sh M3,M4                  # free only M3 + M4 (~45 GB)
#   ./clean.sh "Qwen/Qwen3.5-9B"      # remove by HuggingFace repo ID
#   ./clean.sh --dry-run              # preview without deleting

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${REPO_DIR}/config/models.yaml"
CACHE_DIR=""
DRY_RUN=0
SHOW_FREE=0
MODELS_RAW=""

usage() {
    cat <<'EOF'
Usage: ./clean.sh [OPTIONS] [MODELS]

Standalone host script to remove cached model weights from the HF cache.
Weights are kept by default after each benchmark run so re-runs skip the
re-download; use this to free that disk space.

POSITIONAL:
  MODELS                Model keys (M1,M2,M3,M4) or HuggingFace repo IDs
                        (comma-separated). Default: all cached weights.

OPTIONS:
  --cache-dir <dir>     Override HF cache location (default ./.hf-cache)
  --models <csv>        Comma-separated model keys or repo IDs
                        (same as positional args)
  --dry-run             Show what would be removed without deleting
  --free                Print total disk freed after deletion
  -h, --help            Show this help and exit

EXAMPLES:
  ./clean.sh                        # remove all cached weights (~80 GB)
  ./clean.sh M3,M4                  # free only M3 + M4 (~45 GB)
  ./clean.sh "Qwen/Qwen3.5-9B"      # remove by full HuggingFace repo ID
  ./clean.sh --dry-run              # preview without deleting
  ./clean.sh --free                 # show total freed disk

MODEL KEYS (from config/models.yaml):
  M1 → Qwen/Qwen3.5-9B
  M2 → openai/gpt-oss-20b
  M3 → cyankiwi/Qwen3.8-27B-AWQ-INT4
  M4 → cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit

HF CACHE STRUCTURE:
  <cache_dir>/hub/models--<org>--<name>/
EOF
    exit 0
}

# ── arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-dir)  CACHE_DIR="${2:-}"; shift 2 ;;
        --models)     MODELS_RAW="${2:-}"; shift 2 ;;
        --models=*)   MODELS_RAW="${1#*=}"; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --free)       SHOW_FREE=1; shift ;;
        -h|--help)    usage ;;
        *) MODELS_RAW="${MODELS_RAW:+${MODELS_RAW},}$1"; shift ;;
    esac
done

# Default cache dir: repo-local .hf-cache
if [[ -z "$CACHE_DIR" ]]; then
    CACHE_DIR="$REPO_DIR/.hf-cache"
fi

# ── helpers ──────────────────────────────────────────────────────────────────
log() { echo "[clean] $*"; }
die() { echo "[clean] ERROR: $*" >&2; exit 1; }

# Resolve a model key (M1, M2, ...) or repo ID to the HF dir name.
resolve_model() {
    local raw="$1"

    # Direct repo ID (contains /) — use as-is
    if [[ "$raw" == */* ]]; then
        local org="${raw%%/*}"
        local name="${raw#*/}"
        echo "models--${org//--/--}--${name//--/--}"
        return
    fi

    # Model key (M1, M2, M3, M4) — look up in models.yaml
    local repo_id
    repo_id=$(sed -n "/^  ${raw}:/,/^  [A-Z]/p" "$CONFIG" | grep '^\s*id:' | head -1 | sed 's/.*id:\s*//' | xargs)
    if [[ -z "$repo_id" ]]; then
        die "unknown model key: $raw (expected M1|M2|M3|M4 or a repo ID)"
    fi
    local org="${repo_id%%/*}"
    local name="${repo_id#*/}"
    echo "models--${org//\//--}--${name//\//--}"
}

# ── main ─────────────────────────────────────────────────────────────────────

if [[ -z "$MODELS_RAW" ]]; then
    MODELS_RAW="all"
fi

# Determine the list of models to clean.
declare -a MODEL_KEYS=()
IFS=',' read -ra RAW_LIST <<< "$MODELS_RAW"
for raw in "${RAW_LIST[@]}"; do
    raw="$(echo "$raw" | xargs)"  # trim whitespace
    if [[ "$raw" == "all" ]]; then
        MODEL_KEYS=("all")
        break
    fi
    MODEL_KEYS+=("$raw")
done

# Collect (repo_id, dir_name) pairs
declare -a REPO_PAIRS=()
for key in "${MODEL_KEYS[@]}"; do
    if [[ "$key" == "all" ]]; then
        # Clean everything in the cache
        if [[ -d "$CACHE_DIR/hub" ]]; then
            for d in "$CACHE_DIR/hub/models-"*/; do
                [[ -d "$d" ]] || continue
                REPO_PAIRS+=("$(basename "$d")")
            done
        fi
    else
        read -r dir_name <<< "$(resolve_model "$key")"
        REPO_PAIRS+=("$dir_name")
        log "$key → dir: $dir_name"
    fi
done

# ── delete ────────────────────────────────────────────────────────────────────
if [[ ${#REPO_PAIRS[@]} -eq 0 ]]; then
    log "nothing to clean (cache: $CACHE_DIR)"
    exit 0
fi

TOTAL_FREED=0
for dir_name in "${REPO_PAIRS[@]}"; do
    dir_path="$CACHE_DIR/hub/$dir_name"
    if [[ -d "$dir_path" ]]; then
        dir_size=$(du -sb "$dir_path" 2>/dev/null | awk '{print $1}' || echo 0)
        if [[ "$DRY_RUN" -eq 1 ]]; then
            log "  [dry-run] would remove $dir_name (${dir_size} bytes)"
            TOTAL_FREED=$((TOTAL_FREED + dir_size))
        else
            rm -rf "$dir_path"
            log "  removed $dir_name (${dir_size} bytes)"
            TOTAL_FREED=$((TOTAL_FREED + dir_size))
        fi
    else
        log "  $dir_name not found (skip)"
    fi
done

# ── summary ──────────────────────────────────────────────────────────────────
if [[ "$SHOW_FREE" -eq 1 ]]; then
    gb_freed=$(awk "BEGIN {printf \"%.1f\", $TOTAL_FREED / 1024 / 1024 / 1024}")
    log "total freed: ${TOTAL_FREED} bytes ≈ ${gb_freed} GB"
else
    log "cleaned ${#REPO_PAIRS[@]} entries from $CACHE_DIR"
fi