#!/usr/bin/env bash
# bench.sh — host-side orchestrator for the platform-agnostic GPU inference
# benchmark. Detects the GPU vendor, pulls the pinned vLLM image, and launches
# the container that runs the full model × config matrix.
#
# Usage:
#   ./bench.sh                          # full matrix, auto-detect GPU
#   ./bench.sh --quick                  # M1 only, baseline+kv-fp8, C=1,8
#   ./bench.sh --models M2 --configs baseline
#   ./bench.sh --concurrency 1,8,16
#   ./bench.sh --gpu-index 1            # force a specific physical GPU
#   ./bench.sh --keep-weights           # don't delete weights between models
#   ./bench.sh --vendor amd             # override vendor detection
#   ./bench.sh --dry-run                # print the docker command, don't run
#
# Outputs land in ./results (report.json + report.md + raw bench/telemetry JSON).
set -uo pipefail

# ── constants ────────────────────────────────────────────────────────────────
VLLM_VERSION="${VLLM_VERSION:-v0.28.0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
HF_CACHE_HOST="${HF_CACHE_HOST:-$HOME/.cache/gpu-bench/hf}"
CONTAINER_NAME="gpu-bench"
# Disk gates (GB). HF cache holds one model at a time (weights deleted per
# model), so 30 GB is plenty; the vllm image is ~25 GB so docker root needs
# headroom.
MIN_HF_GB=30
MIN_DOCKER_GB=45

usage() {
    grep '^#   \./bench.sh' "$0" | sed 's/^#   //'
    exit 0
}

# ── arg parsing ──────────────────────────────────────────────────────────────
VENDOR="${GPU_VENDOR:-}"
GPU_INDEX=""
MODELS=""
CONFIGS=""
CONCURRENCY=""
QUICK=0
KEEP_WEIGHTS=0
START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vendor)        VENDOR="${2:-}"; shift 2 ;;
        --vendor=*)      VENDOR="${1#*=}"; shift ;;
        --gpu-index)     GPU_INDEX="${2:-}"; shift 2 ;;
        --gpu-index=*)   GPU_INDEX="${1#*=}"; shift ;;
        --models)        MODELS="${2:-}"; shift 2 ;;
        --models=*)      MODELS="${1#*=}"; shift ;;
        --configs)       CONFIGS="${2:-}"; shift 2 ;;
        --configs=*)     CONFIGS="${1#*=}"; shift ;;
        --concurrency)   CONCURRENCY="${2:-}"; shift 2 ;;
        --concurrency=*) CONCURRENCY="${1#*=}"; shift ;;
        --quick)         QUICK=1; shift ;;
        --keep-weights)  KEEP_WEIGHTS=1; shift ;;
        --start-timeout) START_TIMEOUT="${2:-900}"; shift 2 ;;
        --start-timeout=*) START_TIMEOUT="${1#*=}"; shift ;;
        --results)       RESULTS_DIR="${2:-}"; shift 2 ;;
        --results=*)     RESULTS_DIR="${1#*=}"; shift ;;
        --version)       VLLM_VERSION="${2:-$VLLM_VERSION}"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown arg: $1 (see --help)" >&2; exit 2 ;;
    esac
done

# ── helpers ──────────────────────────────────────────────────────────────────
log()  { echo "[bench] $*"; }
die()  { echo "[bench] ERROR: $*" >&2; exit 1; }

detect_vendor() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "nvidia"
    elif [[ -e /dev/kfd ]]; then
        echo "amd"
    elif command -v xpu-smi >/dev/null 2>&1 && xpu-smi discovery >/dev/null 2>&1; then
        echo "intel"
    elif command -v intel_gpu_top >/dev/null 2>&1; then
        echo "intel"
    else
        echo "unknown"
    fi
}

select_nvidia_gpu() {
    # Only needed for NVIDIA: host has nvidia-smi, so we can pick the largest
    # card and pass --gpus device=N. For AMD the host has no rocm-smi, so the
    # in-container select_gpu() handles selection.
    if [[ -n "$GPU_INDEX" ]]; then
        echo "$GPU_INDEX"; return
    fi
    nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' '{print $1, $2}' \
        | sort -k2 -nr | head -1 | awk '{print $1}'
}

check_disk() {
    local path="$1" need_gb="$2" label="$3"
    [[ -e "$path" ]] || mkdir -p "$path" 2>/dev/null || true
    local avail_kb
    avail_kb=$(df -Pk "$path" 2>/dev/null | awk 'NR==2{print $4}')
    if [[ -z "$avail_kb" ]]; then
        log "WARN: cannot stat disk for $label ($path)"
        return
    fi
    local avail_gb=$(( avail_kb / 1024 / 1024 ))
    if (( avail_gb < need_gb )); then
        log "WARN: $label has ${avail_gb}GB free (< ${need_gb}GB) at $path"
    else
        log "disk: $label ${avail_gb}GB free at $path (need ${need_gb}GB)"
    fi
}

cleanup_stale() {
    # Remove a stale bench container from a crashed prior run.
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    # Remove stale vllm images, but keep the currently pinned tag (avoid a
    # ~25 GB re-pull on every re-run).
    local stale
    stale=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep '^vllm/vllm-openai' | grep -v ":${VLLM_VERSION}$" || true)
    if [[ -n "$stale" ]]; then
        log "removing stale vllm images (keeping :$VLLM_VERSION):"
        echo "$stale" | sed 's/^/    /'
        echo "$stale" | xargs -r docker rmi -f >/dev/null 2>&1 || true
    fi
}

# ── main ─────────────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || die "docker not found"

# 1. Vendor
if [[ -z "$VENDOR" ]]; then
    VENDOR=$(detect_vendor)
fi
case "$VENDOR" in
    nvidia|amd|intel) log "vendor: $VENDOR (vLLM $VLLM_VERSION)" ;;
    *) die "could not detect GPU vendor (nvidia-smi / /dev/kfd / xpu-smi); pass --vendor" ;;
esac

# 2. Image
case "$VENDOR" in
    nvidia) IMAGE="vllm/vllm-openai:${VLLM_VERSION}" ;;
    amd)    IMAGE="vllm/vllm-openai-rocm:${VLLM_VERSION}" ;;
    intel)  IMAGE="vllm/vllm-openai-xpu:${VLLM_VERSION}" ;;
esac
log "image: $IMAGE"

# 3. GPU docker args + host-side selection (NVIDIA only)
GPU_ARGS=()
if [[ "$VENDOR" == "nvidia" ]]; then
    NGPU=$(select_nvidia_gpu)
    [[ -n "$NGPU" ]] || die "no NVIDIA GPU found"
    GPU_INDEX="$NGPU"
    GPU_ARGS=(--gpus "device=$NGPU")
    log "nvidia GPU: device $NGPU"
elif [[ "$VENDOR" == "amd" ]]; then
    # Expose all GPUs; in-container select_gpu() picks by VRAM (or --gpu-index).
    GPU_ARGS=(--device /dev/kfd --device /dev/dri)
    log "amd: exposing /dev/kfd + /dev/dri (in-container selection)"
elif [[ "$VENDOR" == "intel" ]]; then
    GPU_ARGS=(--device /dev/dri)
    log "intel: exposing /dev/dri (best-effort)"
fi

# 4. Disk gates
check_disk "$HF_CACHE_HOST" "$MIN_HF_GB"   "HF cache"
check_disk "/var/lib/docker" "$MIN_DOCKER_GB" "docker root"

# 5. Host ownership
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"

# 6. Cleanup + pull
if [[ "$DRY_RUN" != "1" ]]; then
    cleanup_stale
    log "pulling $IMAGE (no-op if present) ..."
    docker pull "$IMAGE" || log "WARN: docker pull failed (using local image?)"
fi

# 7. Launch
DOCKER_CMD=(docker run --rm --name "$CONTAINER_NAME" --entrypoint bash
    -e GPU_VENDOR="$VENDOR"
    -e GPU_INDEX="${GPU_INDEX:-}"
    -e MODELS="${MODELS:-}"
    -e CONFIGS="${CONFIGS:-}"
    -e CONCURRENCY="${CONCURRENCY:-}"
    -e QUICK="$QUICK"
    -e KEEP_WEIGHTS="$KEEP_WEIGHTS"
    -e HF_HOME=/hf-cache
    -e RESULTS=/results
    -e HOST_UID="$HOST_UID"
    -e HOST_GID="$HOST_GID"
    -e SERVER_START_TIMEOUT="$START_TIMEOUT"
    -v "$REPO_DIR:/bench"
    -v "$HF_CACHE_HOST:/hf-cache"
    -v "$RESULTS_DIR:/results"
    "${GPU_ARGS[@]}"
    "$IMAGE"
    /bench/container/entrypoint.sh)

if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run — would execute:"
    printf '  %q ' "${DOCKER_CMD[@]}"; echo
    exit 0
fi

mkdir -p "$RESULTS_DIR" "$HF_CACHE_HOST"
log "launching container ($CONTAINER_NAME) ..."
log "results → $RESULTS_DIR"
"${DOCKER_CMD[@]}"
RC=$?
echo
log "container exited rc=$RC"
if [[ -f "$RESULTS_DIR/report.md" ]]; then
    log "report: $RESULTS_DIR/report.md"
    echo "────────────────────────────────────────────────────────────"
    cat "$RESULTS_DIR/report.md"
    echo "────────────────────────────────────────────────────────────"
fi
exit "$RC"