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
#   ./bench.sh --delete-weights         # delete weights after each model (old behavior)
#   ./bench.sh --clean [M1,M2,...]      # remove cached weights, then exit
#   ./bench.sh --vendor amd             # override vendor detection
#   ./bench.sh --image <repo:tag>       # override the vLLM image
#   ./bench.sh --cache-dir /big/hf      # HF weights cache directory
#   ./bench.sh --validate               # preflight VRAM-fit check (static+live)
#   ./bench.sh --dry-run                # print the docker command, don't run
#   ./bench.sh --force                  # skip the disk-space gate
#
# Outputs land in ./results/<YYYYMMDD-HHMMSS>_<gpu-slug>/ (one dir per run,
# no collisions; report.json + report.md + raw bench/telemetry JSON inside).
set -uo pipefail

# ── constants ────────────────────────────────────────────────────────────────
VLLM_VERSION="${VLLM_VERSION:-v0.28.0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
# HF model cache defaults to the project folder (keeps it out of the root disk
# / $HOME and co-locates it with the repo). Override with HF_CACHE_HOST to place
# it on a dedicated large volume.
HF_CACHE_HOST="${HF_CACHE_HOST:-$REPO_DIR/.hf-cache}"
CONTAINER_NAME="gpu-bench"
# Disk gates (GB).
#   NEED_MODEL_GB        — HF cache: one model at a time (largest ~24.4 GB,
#                          plus LFS blob→file reconstruction headroom).
#   NEED_DOCKER_GB       — docker root when a fresh image pull is needed
#                          (~25 GB image + container headroom).
#   NEED_DOCKER_PRESENT_GB — docker root headroom when the image is already
#                          present locally.
# When the HF cache and docker root share a filesystem, the requirements are
# summed (the image pull and the weight download compete for the same disk).
NEED_MODEL_GB=30
NEED_DOCKER_GB=35
NEED_DOCKER_PRESENT_GB=15

usage() {
    cat <<'EOF'
Usage: ./bench.sh [OPTIONS]

Host-side orchestrator for the platform-agnostic GPU inference benchmark.
Detects the GPU vendor, pulls the pinned vLLM image, and launches the
container that runs the full model × config matrix.

OPTIONS:
  --quick                 Smoke test: M1 only, baseline+kv-fp8, C=1,8
  --models <csv>          Subset of models, comma-/range-separated
                          (e.g. M2,M4 or M1-M4). Default: all
  --configs <csv>         Subset of configs, comma list (e.g. baseline,kv-fp8)
  --concurrency <csv>     Concurrency sweep, comma list (e.g. 1,8,16). Default: 1,4,8,16
  --gpu-index <N>         Force a specific physical GPU index (auto-pick by VRAM)
  --vendor <amd|nvidia|intel>   Override GPU vendor auto-detection
  --version <tag>         Override the vLLM version (default v0.28.0)
  --image <repo:tag>      Override the full vLLM image name
  --cache-dir <dir>       HF weights cache directory (default ./.hf-cache)
  --results <dir>         Output root (default ./results)
  --start-timeout <sec>   Server health-wait budget (default 900)
  --validate              Preflight VRAM-fit check (static estimate + live probe)
  --clean [M1,M2,...]     Remove cached model weights, then exit (no container)
  --delete-weights        Delete weights after each model (old behavior; off by default
                          so re-runs skip the ~20-25 GB re-download)
  --keep-weights          Deprecated no-op (weights are kept by default now)
  --dry-run               Print the docker command, don't run
  --force                 Skip the disk-space gates
  -h, --help              Show this help and exit

ENVIRONMENT OVERRIDES:
  VLLM_VERSION        vLLM version tag (default v0.28.0)
  GPU_VENDOR          amd | nvidia | intel (auto-detected if unset)
  GPU_INDEX           physical GPU index (same as --gpu-index)
  HF_CACHE_HOST       HF weights cache directory (same as --cache-dir)
  RESULTS_DIR         output root (same as --results)
  SERVER_START_TIMEOUT  server health-wait budget in seconds (default 900)

EXAMPLES:
  ./bench.sh                          # full matrix, auto-detect GPU
  ./bench.sh --quick                  # smoke test (M1, ~5 min)
  ./bench.sh --models M2,M4           # subset of models (comma list)
  ./bench.sh --models M3-M4           # range: M3, M4
  ./bench.sh --validate --models M3,M4  # preflight VRAM-fit check
  ./bench.sh --clean M3,M4            # free M3+M4 weights, then exit
  ./bench.sh --dry-run                # print the docker command, don't run
EOF
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
DELETE_WEIGHTS=0
CLEAN_MODELS=""
VALIDATE=0
START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
DRY_RUN=0
FORCE=0
IMAGE_OVERRIDE=""
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
        --validate)      VALIDATE=1; shift ;;
        --keep-weights)  KEEP_WEIGHTS=1; shift ;;
        --delete-weights) DELETE_WEIGHTS=1; shift ;;
        --clean)         CLEAN_MODELS="${2:-all}"; shift 2 ;;
        --clean=*)       CLEAN_MODELS="${1#*=}"; shift ;;
        --start-timeout) START_TIMEOUT="${2:-900}"; shift 2 ;;
        --start-timeout=*) START_TIMEOUT="${1#*=}"; shift ;;
        --results)       RESULTS_DIR="${2:-}"; shift 2 ;;
        --results=*)     RESULTS_DIR="${1#*=}"; shift ;;
        --version)       VLLM_VERSION="${2:-$VLLM_VERSION}"; shift 2 ;;
        --version=*)     VLLM_VERSION="${1#*=}"; shift ;;
        --image)         IMAGE_OVERRIDE="${2:-}"; shift 2 ;;
        --image=*)       IMAGE_OVERRIDE="${1#*=}"; shift ;;
        --cache-dir)     HF_CACHE_HOST="${2:-}"; shift 2 ;;
        --cache-dir=*)   HF_CACHE_HOST="${1#*=}"; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --force)         FORCE=1; shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown arg: $1 (see --help)" >&2; exit 2 ;;
    esac
done

# Weights are KEPT by default (re-runs skip the re-download), so the disk gate
# must reserve room for the whole weight set. --delete-weights restores the
# old one-model-at-a-time footprint.
if [[ "$DELETE_WEIGHTS" != "1" ]]; then
    NEED_MODEL_GB=85   # M1+M2+M3+M4 ≈ 78.6 GB + headroom (kept on disk)
fi

# ── helpers ──────────────────────────────────────────────────────────────────
log()  { echo "[bench] $*"; }
die()  { echo "[bench] ERROR: $*" >&2; exit 1; }

# ── --clean: remove cached weights and exit (no container, no benchmark) ─────
if [[ -n "$CLEAN_MODELS" ]]; then
    log "cleaning cached weights ($CLEAN_MODELS) from $HF_CACHE_HOST ..."
    bash "$REPO_DIR/clean.sh" --cache-dir "$HF_CACHE_HOST" --models "$CLEAN_MODELS"
    exit $?
fi

detect_vendor() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "nvidia"
    elif command -v xpu-smi >/dev/null 2>&1 && xpu-smi discovery >/dev/null 2>&1; then
        # Checked before /dev/kfd: on mixed AMD+Intel hosts, /dev/kfd exists
        # just because ROCm is installed. A real Intel XPU present wins.
        echo "intel"
    elif [[ -e /dev/kfd ]]; then
        echo "amd"
    elif [[ -d /sys/module/xe ]]; then
        # xe kernel module = discrete Intel GPU (Arc A/B, e.g. Arc B70).
        # intel_gpu_top (intel-gpu-tools) is i915-only and cannot see xe
        # devices, so it is deliberately not used as a signal.
        echo "intel"
    elif lspci 2>/dev/null | grep -qiE 'intel.*(display|vga|3d)'; then
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

# ── disk helpers ─────────────────────────────────────────────────────────────
# Filesystem id (device number) of the FS backing a path — used to detect
# whether two paths share the same disk.
fsid() { stat -c %d "$1" 2>/dev/null || echo ""; }
# Free space (GB) on the FS backing a path.
free_gb() { df -Pk "$1" 2>/dev/null | awk 'NR==2{print int($4/1024/1024)}'; }

# Hard pre-flight gate (unless --force). Ensures enough free space to (a) pull
# the image if not already present, and (b) download the largest model. If the
# HF cache and docker root share a filesystem their requirements are summed.
disk_gate() {
    mkdir -p "$HF_CACHE_HOST" "$RESULTS_DIR" 2>/dev/null || true
    local docker_root
    docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    [[ -n "$docker_root" ]] || docker_root="/var/lib/docker"

    local hf_id docker_id
    hf_id=$(fsid "$HF_CACHE_HOST")
    docker_id=$(fsid "$docker_root")

    local image_present=0
    docker image inspect "$IMAGE" >/dev/null 2>&1 && image_present=1

    local image_need=$NEED_DOCKER_PRESENT_GB
    (( image_present == 0 )) && image_need=$NEED_DOCKER_GB

    local avail
    if [[ -n "$hf_id" && "$hf_id" == "$docker_id" ]]; then
        # Same filesystem: image + model coexist here.
        local total=$(( NEED_MODEL_GB + image_need ))
        avail=$(free_gb "$HF_CACHE_HOST")
        log "disk: HF cache + docker share one FS — need ${total}GB (model ${NEED_MODEL_GB} + docker ${image_need}), have ${avail:-0}GB at $HF_CACHE_HOST"
        if [[ "$FORCE" -eq 0 ]] && { [[ -z "$avail" ]] || (( avail < total )); }; then
            die "not enough disk on shared FS ($HF_CACHE_HOST): need ${total}GB, have ${avail:-0}GB. Free space, set HF_CACHE_HOST to a larger disk, or pass --force."
        fi
    else
        avail=$(free_gb "$HF_CACHE_HOST")
        log "disk: HF cache ${avail:-0}GB free at $HF_CACHE_HOST (need ${NEED_MODEL_GB}GB)"
        if [[ "$FORCE" -eq 0 ]] && { [[ -z "$avail" ]] || (( avail < NEED_MODEL_GB )); }; then
            die "not enough disk for HF cache ($HF_CACHE_HOST): need ${NEED_MODEL_GB}GB, have ${avail:-0}GB. Free space, set HF_CACHE_HOST to a larger disk, or pass --force."
        fi
        local dk; dk=$(free_gb "$docker_root")
        log "disk: docker root ${dk:-0}GB free at $docker_root (need ${image_need}GB)"
        if [[ "$FORCE" -eq 0 ]] && { [[ -z "$dk" ]] || (( dk < image_need )); }; then
            die "not enough disk for docker root ($docker_root): need ${image_need}GB, have ${dk:-0}GB. Free space or pass --force."
        fi
    fi
}

# Re-check the HF-cache FS *after* the image pull (the pull can consume a
# shared disk). Aborts if the largest model no longer fits.
post_pull_gate() {
    local avail; avail=$(free_gb "$HF_CACHE_HOST")
    if [[ "$FORCE" -eq 0 ]] && { [[ -z "$avail" ]] || (( avail < NEED_MODEL_GB )); }; then
        die "post-pull: HF cache ($HF_CACHE_HOST) has ${avail:-0}GB free (< ${NEED_MODEL_GB}GB for the largest model). The image pull likely consumed the shared disk. Free space, point HF_CACHE_HOST at a larger disk, or pass --force."
    fi
    log "post-pull disk: HF cache ${avail:-0}GB free (need ${NEED_MODEL_GB}GB)"
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

# Per-run output dir id: host timestamp (passed to the container; run_matrix.py
# appends the GPU-model slug after in-container GPU selection).
RUN_ID="$(date +%Y%m%d-%H%M%S)"

# 1. Vendor
if [[ -z "$VENDOR" ]]; then
    VENDOR=$(detect_vendor)
fi
case "$VENDOR" in
    nvidia|amd|intel) log "vendor: $VENDOR (vLLM $VLLM_VERSION)" ;;
    *) die "could not detect GPU vendor (nvidia-smi / /dev/kfd / xpu-smi / xe module); pass --vendor" ;;
esac

# 2. Image
case "$VENDOR" in
    nvidia) IMAGE="vllm/vllm-openai:${VLLM_VERSION}" ;;
    amd)    IMAGE="vllm/vllm-openai-rocm:${VLLM_VERSION}" ;;
    intel)  IMAGE="vllm/vllm-openai-xpu:${VLLM_VERSION}" ;;
esac
[[ -n "$IMAGE_OVERRIDE" ]] && IMAGE="$IMAGE_OVERRIDE"
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
    # docker --device wants real device nodes, not the /dev/dri directory.
    # Pass every DRI node explicitly; the in-container select_gpu() then
    # picks the dGPU by VRAM via torch.xpu (xe driver: renderD12x nodes).
    INTEL_NODES=()
    for n in /dev/dri/renderD* /dev/dri/card*; do
        [[ -c "$n" ]] && INTEL_NODES+=("$n")
    done
    if [[ ${#INTEL_NODES[@]} -eq 0 ]]; then
        die "intel: no /dev/dri/renderD* device nodes — is the xe kernel module loaded? (lsmod | grep xe)"
    fi
    GPU_ARGS=()
    for n in "${INTEL_NODES[@]}"; do
        GPU_ARGS+=(--device "$n")
    done
    # Standard for Intel GPU containers; only added when the group exists
    # (docker rejects unknown group names).
    for g in video render; do
        getent group "$g" >/dev/null 2>&1 && GPU_ARGS+=(--group-add "$g")
    done
    log "intel: exposing ${INTEL_NODES[*]}"
fi

# 4. Pre-flight disk gate (hard unless --force)
disk_gate

# 5. Host ownership + host-side metadata for environment.json
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"
HOST_OS="$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-}" || uname -s)"
DOCKER_VERSION="$(docker version --format '{{.Server.Version}}' 2>/dev/null)"
DOCKER_VERSION="${DOCKER_VERSION:-unknown}"
log "host os: ${HOST_OS:-?} · docker: $DOCKER_VERSION"

# 6. Cleanup + pull + post-pull disk gate
if [[ "$DRY_RUN" != "1" ]]; then
    cleanup_stale
    log "pulling $IMAGE (no-op if present) ..."
    if ! docker pull "$IMAGE"; then
        if docker image inspect "$IMAGE" >/dev/null 2>&1; then
            log "WARN: docker pull failed but image present locally — using it"
        else
            die "docker pull failed and no local $IMAGE — check network / registry access"
        fi
    fi
    post_pull_gate
fi
# Image ID (digest) after the pull so first runs capture it.
IMAGE_DIGEST="$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || echo "")"

# 7. Launch
DOCKER_CMD=(docker run --rm --name "$CONTAINER_NAME" --entrypoint bash
    -e GPU_VENDOR="$VENDOR"
    -e GPU_INDEX="${GPU_INDEX:-}"
    -e MODELS="${MODELS:-}"
    -e CONFIGS="${CONFIGS:-}"
    -e CONCURRENCY="${CONCURRENCY:-}"
    -e QUICK="$QUICK"
    -e KEEP_WEIGHTS="$KEEP_WEIGHTS"
    -e DELETE_WEIGHTS="$DELETE_WEIGHTS"
    -e VALIDATE="$VALIDATE"
    -e HF_HOME=/hf-cache
    -e RESULTS=/results
    -e RUN_ID="$RUN_ID"
    -e HOST_OS="$HOST_OS"
    -e DOCKER_VERSION="$DOCKER_VERSION"
    -e IMAGE="$IMAGE"
    -e IMAGE_DIGEST="$IMAGE_DIGEST"
    -e HOST_UID="$HOST_UID"
    -e HOST_GID="$HOST_GID"
    -e SERVER_START_TIMEOUT="$START_TIMEOUT"
    # The 32 GB matrix lives at the edge of VRAM: on tight cells (M3/M4) the
    # caching allocator fragments the reserved pool and the last ~1 GiB
    # (CUDA-graph capture / profiling activation) OOMs. Expandable segments
    # let PyTorch grow segments instead of stalling on fragmentation (the
    # remedy in torch's own OOM message; recommended for tight ROCm fits).
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
# Remove any stale .latest from a previous run so a failed new run can't
# be misreported as the old one (run_matrix.py rewrites it on success).
rm -f "$RESULTS_DIR/.latest"
log "launching container ($CONTAINER_NAME) ..."
log "results → $RESULTS_DIR"
"${DOCKER_CMD[@]}"
RC=$?
echo
log "container exited rc=$RC"
if [[ "$VALIDATE" == "1" ]]; then
    # Validate mode: fit_report lives in validate_<ts>_<gpu>/ under results.
    VAL_DIR="$(ls -dt "$RESULTS_DIR"/validate_* 2>/dev/null | head -1 || true)"
    if [[ -n "$VAL_DIR" && -f "$VAL_DIR/fit_report.md" ]]; then
        log "fit report: $VAL_DIR/fit_report.md"
        echo "───────────────────────────────────────────────────────────────"
        cat "$VAL_DIR/fit_report.md"
        echo "───────────────────────────────────────────────────────────────"
    else
        log "fit report: not found (validate dir not written to $RESULTS_DIR)"
    fi
    exit "$RC"
fi
# Locate this run's dir (container writes the run-id basename to
# $RESULTS_DIR/.latest; fall back to legacy top-level report.md).
LATEST="$(cat "$RESULTS_DIR/.latest" 2>/dev/null || true)"
RUN_DIR=""
if [[ -n "$LATEST" && -d "$RESULTS_DIR/$LATEST" && -f "$RESULTS_DIR/$LATEST/report.md" ]]; then
    RUN_DIR="$RESULTS_DIR/$LATEST"
elif [[ -f "$RESULTS_DIR/report.md" ]]; then
    RUN_DIR="$RESULTS_DIR"
fi
if [[ -n "$RUN_DIR" ]]; then
    log "report: $RUN_DIR/report.md"
    echo "────────────────────────────────────────────────────────────"
    cat "$RUN_DIR/report.md"
    echo "────────────────────────────────────────────────────────────"
fi
exit "$RC"