#!/usr/bin/env bash
# clean.sh — standalone script to remove cached model weights from the HF cache.
#
# Usage:
#   ./clean.sh                    # remove ALL cached weights from default cache
#   ./clean.sh M1,M2,M3,M4        # remove specific models (keys from config/models.yaml)
#   ./clean.sh "Qwen/Qwen3.5-9B"  # remove by HuggingFace repo ID (direct)
#
# Options:
#   --cache-dir <dir>   override HF cache location (default: .hf-cache in repo root)
#   --dry-run           show what would be removed, do not delete
#   --free              print total disk freed after deletion
#
# The HF cache structure is:
#   <cache_dir>/hub/models--<org>--<name>/
#
# Weights are downloaded by vllm/snapshot_download and cached here. Removing
# them does NOT affect the vLLM Docker image, only cached checkpoints.
#
# Example: speed up re-runs by keeping weights (default), then clean when done
#   ./bench.sh --models M3,M4   # runs and caches M3+M4 weights
#   ./clean.sh M3,M4            # frees ~45 GB

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${REPO_DIR}/config/models.yaml"
CACHE_DIR=""
DRY_RUN=0
SHOW_FREE=0
MODELS_RAW=""

# ── arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-dir)  CACHE_DIR="${2:-}"; shift 2 ;;
        --models)     MODELS_RAW="${2:-}"; shift 2 ;;
        --models=*)   MODELS_RAW="${1#*=}"; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --free)       SHOW_FREE=1; shift ;;
        -h|--help)
            echo "$0"
            grep '^#   \./clean.sh\|^#   \./clean\.sh\|^#   --' "$0" | sed 's/^#   //'
            echo
            echo "Model keys (from $CONFIG): M1 M2 M3 M4"
            exit 0
            ;;
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

# Resolve a model key (M1, M2, ...) or repo ID to the HF repo ID and dir name.
# Returns "repo_id dir_name" space-separated.
resolve_model() {
    local raw="$1"

    # Direct repo ID (contains /) — use as-is
    if [[ "$raw" == */* ]]; then
        local org="${raw%%/*}"
        local name="${raw#*/}"
        echo "$raw models--${org//--/--}--${name//--/--}"
        return
    fi

    # Model key (M1, M2, M3, M4) — look up in models.yaml
    # Simple grep: look for 'id: <repo>' under the matching model block
    # Pattern: model key followed by id: line
    local repo_id
    repo_id=$(sed -n "/^  ${raw}:/,/^  [A-Z]/p" "$CONFIG" | grep '^\s*id:' | head -1 | sed 's/.*id:\s*//' | xargs)
    if [[ -z "$repo_id" ]]; then
        die "unknown model key: $raw (expected M1|M2|M3|M4 or a repo ID)"
    fi
    local org="${repo_id%%/*}"
    local name="${repo_id#*/}"
    # Handle repo names with hyphens (replace -- in org with --)
    local dir_name="models--${org//\//--}--${name//\//--}"
    echo "$repo_id $dir_name"
}

# ── main ─────────────────────────────────────────────────────────────────────

if [[ -z "$MODELS_RAW" ]]; then
    MODELS_RAW="all"
fi

# Collect positional args (bare repo IDs / keys) — comma or space separated.
# They are appended to MODELS_RAW if any were given.

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
        read -r repo_id dir_name <<< "$(resolve_model "$key")"
        REPO_PAIRS+=("$dir_name")
        log "$key → $repo_id (dir: $dir_name)"
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
