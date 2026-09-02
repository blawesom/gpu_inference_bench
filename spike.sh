#!/usr/bin/env bash
#
# spike.sh — vLLM v0.28.0 verification spike (milestone 2 of PLAN.md)
#
# Runs on ONE machine with a GPU (NVIDIA / AMD / Intel). Verifies:
#   S1  vLLM version + how to invoke the benchmark module
#       (`vllm bench serve` vs `python3 -m vllm.benchmark.serve`) +
#       flag availability (--save-result, --result-filename, --num-warmups,
#       --percentile-metrics, --ignore-eos, --max-concurrency)
#   S2  baseline: serve Qwen3.5-4B, one small bench run, capture the
#       result JSON schema
#   S3  `--kv-cache-dtype fp8`: does the server start? (error captured if not)
#   S4  MTP speculative decoding on gpt-oss-20b (tries candidate flag values;
#       the vLLM error message lists supported method names if the guess fails)
#   S5  telemetry tools available in-container (nvidia-smi / rocm-smi /
#       intel_gpu_top) and whether a power field exists
#
# Usage:
#   ./spike.sh                    # full spike (~25 GB download, ~30-45 min)
#   ./spike.sh --skip-gpt-oss     # skip S4 (saves ~14 GB)
#   ./spike.sh --gpu-index 1      # AMD: force the in-container HIP index
#   IMAGE_TAG=v0.27.1 ./spike.sh  # try a different pinned tag
#
# At launch it cleans previous-run state: orphaned spike container, model
# cache ($HF_CACHE/hub), and stale vllm spike images (current tag is kept
# to avoid a ~25 GB re-pull).
#
# Outputs: ./spike-out/  (step logs, bench JSON, summary.txt)

set -euo pipefail

SKIP_GPT_OSS=0
IMAGE_TAG="${IMAGE_TAG:-v0.28.0}"
FORCE_GPU_IDX=""
OUT_DIR="$(pwd)/spike-out"
HF_CACHE="$(pwd)/spike-hf-cache"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-gpt-oss) SKIP_GPT_OSS=1; shift ;;
    --image-tag)    IMAGE_TAG="$2"; shift 2 ;;
    --gpu-index)    FORCE_GPU_IDX="$2"; shift 2 ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[spike] $*" >&2; }

# ── vendor detection ─────────────────────────────────────────────────────────
detect_vendor() {
  if command -v lspci >/dev/null 2>&1; then
    lspci 2>/dev/null | grep -qiE 'nvidia' && { echo nvidia; return; }
    lspci 2>/dev/null | grep -qiE 'vga|display|3d' | grep -qiE 'amd|radeon' && { echo amd; return; }
    lspci 2>/dev/null | grep -qiE 'vga|display|3d' | grep -qiE 'intel' && { echo intel; return; }
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then echo nvidia; return; fi
  [[ -e /dev/kfd ]] && { echo amd; return; }
  [[ -e /dev/dri ]] && { echo intel; return; }
  return 1
}

VENDOR="$(detect_vendor)" || { log "ERROR: no supported GPU detected (lspci / nvidia-smi / /dev/dri)"; exit 1; }
log "vendor: $VENDOR"

# Pick the GPU with the most VRAM when multiple devices are present.
# Output: GPU_IDX (index for the vendor env var), GPU_DESC, GPU_VRAM_MB
MIN_VRAM_MB=24576   # 24 GB escape gate (intel exempt: no reliable host query)
pick_gpu() {
  local vendor="$1"
  GPU_IDX=0; GPU_VRAM_MB=0; GPU_DESC="unknown"
  case "$vendor" in
    nvidia)
      local idx name vram best=-1
      while IFS=, read -r idx name vram _; do
        idx=${idx// /}; name=${name// /}; vram=${vram// /}
        [[ $vram =~ ^[0-9]+$ ]] || continue
        [[ -n "$FORCE_GPU_IDX" && "$idx" != "$FORCE_GPU_IDX" ]] && continue
        if [[ -n "$FORCE_GPU_IDX" ]]; then
          best=$vram; GPU_IDX=$idx; GPU_DESC="${name} ($vram MiB, forced)"
        else
          (( vram > best )) || continue
          best=$vram; GPU_IDX=$idx; GPU_DESC="${name} ($vram MiB)"
        fi
      done < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null)
      GPU_VRAM_MB=$best
      ;;
    amd)
      # Host side only sanity-gates that the machine has a big-enough card.
      # The actual GPU index is chosen INSIDE the container (step S0), where
      # rocm-smi's GPU index IS the HIP index — host-side sysfs card numbers
      # do not reliably map to HIP indices on multi-card systems.
      local rocm_bin="" f vram best=-1
      if command -v rocm-smi >/dev/null 2>&1; then rocm_bin=rocm-smi
      elif [[ -x /opt/rocm/bin/rocm-smi ]]; then rocm_bin=/opt/rocm/bin/rocm-smi
      fi
      if [[ -n "$rocm_bin" ]]; then
        while IFS=, read -r _ vram _; do
          vram=${vram//\"/}; vram=${vram// /}
          [[ $vram =~ ^[0-9]+$ ]] || continue
          (( vram > best )) || continue
          best=$(( vram / 1048576 ))
        done < <("$rocm_bin" --showmeminfo vram --csv 2>/dev/null | tail -n +2)
      fi
      if (( best > 0 )); then
        GPU_VRAM_MB=$best; GPU_DESC="host rocm-smi max ${best} MiB (index chosen in-container)"
      else
        for f in /sys/class/drm/card*/device/mem_info_vram_total; do
          [[ -r $f ]] || continue
          vram=$(<"$f")
          [[ $vram =~ ^[0-9]+$ ]] || continue
          (( vram > best )) || continue
          best=$(( vram / 1048576 ))
        done
        GPU_VRAM_MB=$best; GPU_DESC="host sysfs max ${best} MiB (index chosen in-container)"
      fi
      GPU_IDX=-1
      ;;
    intel)
      local n; n=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
      GPU_DESC="render node 0 ($n render device(s))"
      GPU_VRAM_MB=999999   # no reliable host-side VRAM query; exempt from gate
      ;;
  esac

  if [[ "$vendor" != "intel" ]] && (( GPU_VRAM_MB < MIN_VRAM_MB )); then
    log "ERROR: selected GPU has ${GPU_VRAM_MB} MiB VRAM (< 24 GB minimum) — aborting."
    log "GPU: $GPU_DESC. This benchmark targets 32-40 GB GPUs."
    exit 1
  fi
}
pick_gpu "$VENDOR"
if (( GPU_IDX >= 0 )); then log "gpu: index $GPU_IDX — $GPU_DESC"; else log "gpu: $GPU_DESC"; fi

case "$VENDOR" in
  nvidia) IMAGE="vllm/vllm-openai:$IMAGE_TAG";     GPU_FLAGS=(--gpus "device=$GPU_IDX") ; GPU_ENV_ARGS=(-e "CUDA_VISIBLE_DEVICES=$GPU_IDX") ;;
  amd)    IMAGE="vllm/vllm-openai-rocm:$IMAGE_TAG"; GPU_FLAGS=(--device=/dev/kfd --device=/dev/dri --group-add=video --security-opt seccomp=unconfined); GPU_ENV_ARGS=() ;;  # index chosen in-container (S0); array stays empty so no dangling -e
  intel)  IMAGE="vllm/vllm-openai-xpu:$IMAGE_TAG";  GPU_FLAGS=(--device=/dev/dri --group-add=video); GPU_ENV_ARGS=(-e "ONEAPI_DEVICE_SELECTOR=0") ;;
esac
log "image: $IMAGE"

command -v docker >/dev/null 2>&1 || { log "ERROR: docker not found"; exit 1; }

# ── clean previous-run state (known starting point + honest disk check) ──────
log "cleaning previous-run state..."
docker rm -f gpu-bench-spike >/dev/null 2>&1 || true            # orphaned container (frees the GPU)
mkdir -p "$HF_CACHE"
rm -rf "$HF_CACHE/hub"                                          # stale/corrupted model blobs → clean re-download
for img in $(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -E '^vllm/vllm-openai(-rocm|-xpu)?:'); do
  [[ "$img" == "$IMAGE" ]] && continue                          # keep current tag (avoids ~25 GB re-pull)
  docker rmi "$img" >/dev/null 2>&1 && log "  removed old image $img" || true
done
df_free_gb() { df -BG "$1" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}'; }
NEED_HF_GB=30; NEED_DOCKER_GB=45
(( SKIP_GPT_OSS )) || NEED_HF_GB=50                    # +gpt-oss-20b 13.8 GB + rebuild headroom
docker image inspect "$IMAGE" >/dev/null 2>&1 && NEED_DOCKER_GB=10
HF_FREE_GB=$(df_free_gb "$(pwd)");            [[ $HF_FREE_GB =~ ^[0-9]+$ ]] || HF_FREE_GB=0
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
DOCKER_FREE_GB=$(df_free_gb "$DOCKER_ROOT");   [[ $DOCKER_FREE_GB =~ ^[0-9]+$ ]] || DOCKER_FREE_GB=0
(( HF_FREE_GB >= NEED_HF_GB )) || { log "ERROR: need ≥ ${NEED_HF_GB} GB free for model cache on $(pwd) (got ${HF_FREE_GB} GB)"; exit 1; }
(( DOCKER_FREE_GB >= NEED_DOCKER_GB )) || { log "ERROR: need ≥ ${NEED_DOCKER_GB} GB free for docker images at $DOCKER_ROOT (got ${DOCKER_FREE_GB} GB)"; exit 1; }
log "disk: ${HF_FREE_GB} GB free (models, need ${NEED_HF_GB}); ${DOCKER_FREE_GB} GB free (docker, need ${NEED_DOCKER_GB})"
mkdir -p "$OUT_DIR" "$HF_CACHE"
log "pulling image..."
docker pull "$IMAGE"

log "running spike container (~30-45 min). Ctrl-C kills container only."
docker run -i --rm \
  --name gpu-bench-spike \
  --init \
  --entrypoint bash \
  "${GPU_FLAGS[@]}" \
  --shm-size 16g \
  -v "$OUT_DIR:/out" \
  -v "$HF_CACHE:/hf-cache" \
  -e HF_HOME=/hf-cache \
  "${GPU_ENV_ARGS[@]}" \
  -e EXPECTED_VRAM_MB="$GPU_VRAM_MB" \
  -e EXPECTED_GPU_IDX="$GPU_IDX" \
  -e MIN_VRAM_MB="$MIN_VRAM_MB" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -e SKIP_GPT_OSS="$SKIP_GPT_OSS" \
  -e GPU_VENDOR="$VENDOR" \
  "$IMAGE" -s <<'EOSPIKE'
set -u
OUT=/out
cd "$OUT"
mkdir -p "$OUT"

# fresh summary + logs for this run (spike-out persists across runs on the host)
rm -f "$OUT"/summary.txt "$OUT"/s1-* "$OUT"/s2-* "$OUT"/s3-* "$OUT"/s4-* "$OUT"/s5-* 2>/dev/null || true

# every step writes logs to /out; STATUS lines in summary.txt are greppable
step_begin() { echo "==== S$1 $2 ====" | tee -a summary.txt; }
step_ok()    { echo "STATUS: S$1 $2: OK — ${3:-}" | tee -a summary.txt; }
step_fail()  { echo "STATUS: S$1 $2: FAILED — ${3:-}" | tee -a summary.txt; }

wait_health() { # $1=port $2=timeout_s
  for _ in $(seq 1 $(( $2 / 5 ))); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    [[ -n "${SRV_PID:-}" ]] && ! kill -0 "$SRV_PID" 2>/dev/null && return 1   # server already dead
    sleep 5
  done
  return 1
}

# ── S0: AMD — pick the GPU inside the container (rocm-smi idx == HIP idx) ──
# Host-side sysfs card numbers do not reliably map to HIP indices on
# multi-card AMD systems, so the index is chosen here, where vLLM will run.
if [[ "${GPU_VENDOR:-}" == "amd" ]]; then
  step_begin 0 "AMD GPU selection (in-container)"
  lines=$( { rocm-smi --showmeminfo vram 2>/dev/null || true; } \
    | sed -nE 's/^[[:space:]]*GPU\[([0-9]+)\][^:]*:[[:space:]]*VRAM Total Memory \(B\): ([0-9]+).*/\1 \2/p' )
  [[ -n "$lines" ]] || lines=$( { rocm-smi --showmeminfo vram --csv 2>/dev/null || true; } \
    | tail -n +2 | tr -d '"' | awk -F, 'NF>=2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {print $1" "$2}' )
  sel_idx=""; sel_mb=0; best_mb=-1
  while read -r idx vram_b; do
    [[ ${idx:-} =~ ^[0-9]+$ && ${vram_b:-} =~ ^[0-9]+$ ]] || continue
    mb=$(( vram_b / 1048576 ))
    if [[ -n "${FORCE_GPU_IDX:-}" ]]; then
      [[ "$idx" == "$FORCE_GPU_IDX" ]] || continue
      sel_idx=$idx; sel_mb=$mb; break
    fi
    (( mb > best_mb )) || continue
    best_mb=$mb; sel_idx=$idx; sel_mb=$mb
  done <<< "$lines"
  if [[ -z "$sel_idx" ]]; then
    echo "STATUS: S0 GPU selection: FAILED — no matching GPU in container (forced: ${FORCE_GPU_IDX:-none}; available: $(echo $lines | awk '{printf "%s", $1 " "}'))" | tee -a summary.txt
    exit 1
  fi
  if (( sel_mb < ${MIN_VRAM_MB:-24576} )); then
    echo "STATUS: S0 GPU selection: FAILED — gpu$sel_idx has ${sel_mb} MiB (< ${MIN_VRAM_MB} MiB minimum)" | tee -a summary.txt
    exit 1
  fi
  export HIP_VISIBLE_DEVICES=$sel_idx
  GPU_IDX=$sel_idx; GPU_VRAM_MB=$sel_mb; GPU_DESC="in-container gpu$sel_idx (${sel_mb} MiB)"
  EXPECTED_VRAM_MB=$sel_mb; EXPECTED_GPU_IDX=$sel_idx
  echo "STATUS: S0 GPU selection: OK — gpu$sel_idx (${sel_mb} MiB), HIP_VISIBLE_DEVICES=$sel_idx" | tee -a summary.txt
fi
kill_server() {
  [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null
  wait 2>/dev/null || true
  SRV_PID=""
  sleep 8
}

# ── S1: version + benchmark command discovery ────────────────────────────────
step_begin 1 "vllm version + bench command discovery"
{
  echo "--- vllm --version ---"
  vllm --version 2>&1 || true
  echo "--- python / torch ---"
  python3 -c "import torch; print('torch', torch.__version__)" 2>&1 || true
  echo "--- which vllm ---"
  command -v vllm || true
  echo "--- visible GPU (runtime verification of host-side selection) ---"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    || rocm-smi --showproductname --showmeminfo vram 2>/dev/null \
    || /opt/rocm/bin/rocm-smi --showproductname --showmeminfo vram 2>/dev/null \
    || { echo "(no host gpu tool); /dev/dri:"; ls -la /dev/dri 2>/dev/null; }
  # hard check: what vLLM will actually see (device 0 after visibility
  # filtering). torch is authoritative; smi tools are fallback only (they
  # list ALL cards and ignore *_VISIBLE_DEVICES).
  visible_vram_mb=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024*1024))" 2>/dev/null) || visible_vram_mb=""
  if ! [[ ${visible_vram_mb:-} =~ ^[0-9]+$ ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      visible_vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)   # MiB
    elif command -v rocm-smi >/dev/null 2>&1; then
      vram_b=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -1 | cut -d, -f2 | tr -d '" ')
      [[ $vram_b =~ ^[0-9]+$ ]] && visible_vram_mb=$(( vram_b / 1048576 ))                                        # B -> MiB
    elif [[ -x /opt/rocm/bin/rocm-smi ]]; then
      vram_b=$(/opt/rocm/bin/rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -1 | cut -d, -f2 | tr -d '" ')
      [[ $vram_b =~ ^[0-9]+$ ]] && visible_vram_mb=$(( vram_b / 1048576 ))
    fi
  fi
  if [[ "${visible_vram_mb:-}" =~ ^[0-9]+$ ]] && (( visible_vram_mb > 0 )); then
    expected="${EXPECTED_VRAM_MB:-0}"
    if (( expected > 0 )); then
      lo=$(( expected * 90 / 100 )); hi=$(( expected * 110 / 100 ))
      if (( visible_vram_mb >= lo && visible_vram_mb <= hi )); then
        step_ok 1 "visible GPU" "${visible_vram_mb} MiB matches host selection (${expected} MiB @ idx ${EXPECTED_GPU_IDX:-?})"
      else
        step_fail 1 "visible GPU" "MISMATCH: container sees ${visible_vram_mb} MiB, host picked ${expected} MiB (idx ${EXPECTED_GPU_IDX:-?}) — card/HIP index mismatch; rerun with --gpu-index"
      fi
    else
      step_ok 1 "visible GPU" "${visible_vram_mb} MiB (no host expectation to verify against)"
    fi
  else
    step_fail 1 "visible GPU" "could not query visible GPU VRAM in container"
  fi
  echo "--- try: vllm bench serve --help ---"
  vllm bench serve --help 2>&1 | head -40
  echo "--- try: python3 -m vllm.benchmark.serve --help ---"
  python3 -m vllm.benchmark.serve --help 2>&1 | head -40
} > s1-discovery.log 2>&1

# pick the working invocation (v0.28 help starts with lowercase "usage:")
if grep -qi "usage" <(vllm bench serve --help 2>&1); then
  BENCH="vllm bench serve"
  step_ok 1 "bench invocation" "vllm bench serve"
elif grep -qi "usage" <(python3 -m vllm.bench.serve --help 2>&1); then
  BENCH="python3 -m vllm.bench.serve"
  step_ok 1 "bench invocation" "python3 -m vllm.bench.serve"
else
  BENCH="vllm bench serve"
  step_fail 1 "bench invocation" "neither worked; see s1-discovery.log"
fi

# v0.28 help is grouped by config; --help=all lists every flag (fallback: --help)
$BENCH --help=all > s1-bench-help.log 2>&1 || $BENCH --help > s1-bench-help.log 2>&1 || true
for f in save-result result-filename num-warmups percentile-metrics ignore-eos max-concurrency random-input-len seed temperature; do
  if grep -q -- "--$f" s1-bench-help.log; then step_ok 1 "flag --$f"; else step_fail 1 "flag --$f" "missing (see s1-bench-help.log)"; fi
done
grep -E -- "--kv-cache-dtype|--speculative-config" <(vllm serve --help=all 2>&1 || vllm serve --help 2>&1) > s1-serve-flags.log 2>&1 || true
cat s1-serve-flags.log

# ── S2: baseline serve + bench (small model) ─────────────────────────────────
step_begin 2 "baseline: Qwen3.5-4B + bench run"
vllm serve Qwen/Qwen3.5-4B \
  --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.7 \
  --trust-remote-code > s2-serve.log 2>&1 &
SRV_PID=$!
if wait_health 8000 600; then
  step_ok 2 "server up (Qwen3.5-4B)" "health OK on :8000"
  $BENCH \
    --host 127.0.0.1 --port 8000 \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model Qwen/Qwen3.5-4B \
    --dataset-name random \
    --random-input-len 512 --random-output-len 256 \
    --num-prompts 10 --max-concurrency 1 \
    --seed 42 --temperature 0 --ignore-eos \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
    --save-result --result-filename s2-bench-baseline.json \
    > s2-bench.log 2>&1 && step_ok 2 "bench run" "10 prompts @ C=1" \
    || step_fail 2 "bench run" "see s2-bench.log"
  if [[ -s s2-bench-baseline.json ]]; then
    python3 -c "
import json
d = json.load(open('s2-bench-baseline.json'))
print('result JSON keys:', sorted(d.keys()))
" > s2-json-schema.log 2>&1
    step_ok 2 "result JSON schema" "see s2-json-schema.log"
  else
    step_fail 2 "result JSON" "not produced (check --result-filename in s1-bench-help.log)"
  fi
else
  step_fail 2 "server start" "see s2-serve.log"; tail -25 s2-serve.log >> summary.txt
fi
kill_server

# ── S3: kv-cache-dtype fp8 ────────────────────────────────────────────────────
step_begin 3 "kv-cache-dtype fp8 (Qwen3.5-4B)"
vllm serve Qwen/Qwen3.5-4B \
  --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.7 \
  --kv-cache-dtype fp8 --trust-remote-code > s3-serve.log 2>&1 &
SRV_PID=$!
if wait_health 8000 300; then
  step_ok 3 "server up with fp8 KV" "health OK on :8000"
  $BENCH \
    --host 127.0.0.1 --port 8000 --backend openai-chat --endpoint /v1/chat/completions \
    --model Qwen/Qwen3.5-4B --dataset-name random \
    --random-input-len 512 --random-output-len 256 \
    --num-prompts 5 --max-concurrency 1 \
    --seed 42 --temperature 0 --ignore-eos \
    > s3-bench.log 2>&1 && step_ok 3 "bench run with fp8 KV" \
    || step_fail 3 "bench run" "see s3-bench.log"
else
  step_fail 3 "server start with fp8 KV" "expected on some backends — reason:"
  grep -iE "fp8|kv.cache|not.support|error" s3-serve.log | tail -8 >> summary.txt
fi
kill_server

# ── S4: MTP speculative decoding on gpt-oss-20b ──────────────────────────────
if [[ "$SKIP_GPT_OSS" == "1" ]]; then
  step_begin 4 "MTP spec decode (gpt-oss-20b)"; step_fail 4 "skipped" "--skip-gpt-oss"
else
  step_begin 4 "MTP spec decode (gpt-oss-20b)"
  for METHOD in mtp gpt_oss_mtp ngram; do
    # Use the dotted config syntax (explicitly supported by vLLM's
    # FlexibleArgumentParser) instead of one big JSON arg — more robust
    # across CLI wrappers/backends. Resolved command is logged for post-mortem.
    S4_CMD=(vllm serve openai/gpt-oss-20b
            --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85
            --speculative-config.method "$METHOD"
            --speculative-config.num_speculative_tokens 1
            --trust-remote-code)
    { printf '%q ' "${S4_CMD[@]}"; echo; } > "s4-serve-${METHOD}.log"
    "${S4_CMD[@]}" >> "s4-serve-${METHOD}.log" 2>&1 &
    SRV_PID=$!
    sleep 3   # let argparse errors land before the health poll starts
    if wait_health 8000 900; then
      step_ok 4 "server up with spec method '$METHOD'" "health OK on :8000"
      $BENCH \
        --host 127.0.0.1 --port 8000 --backend openai-chat --endpoint /v1/chat/completions \
        --model openai/gpt-oss-20b --dataset-name random \
        --random-input-len 512 --random-output-len 256 \
        --num-prompts 5 --max-concurrency 1 \
        --seed 42 --temperature 0 --ignore-eos \
        > "s4-bench-${METHOD}.log" 2>&1 && step_ok 4 "bench run (method=$METHOD)" \
        || step_fail 4 "bench run" "see s4-bench-${METHOD}.log"
      break   # found a working method — done
    else
      step_fail 4 "method '$METHOD'" "server failed to start"
      grep -iE "speculat|method|supported|error" "s4-serve-${METHOD}.log" | tail -6 >> summary.txt
    fi
    kill_server
  done
  kill_server
fi

# ── S5: telemetry tools ───────────────────────────────────────────────────────
step_begin 5 "telemetry tools"if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi > s5-nvidia.log 2>&1; then
  step_ok 5 "nvidia-smi"
  grep -qiE "power" s5-nvidia.log && step_ok 5 "power field" || step_fail 5 "power field" "n/a in output"
elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi > s5-rocm.log 2>&1; then
  step_ok 5 "rocm-smi"
  grep -qiE "power" s5-rocm.log && step_ok 5 "power field" || step_fail 5 "power field" "n/a in output"
elif command -v intel_gpu_top >/dev/null 2>&1 && timeout 10 intel_gpu_top --json > s5-intel.log 2>&1; then
  step_ok 5 "intel_gpu_top"
  grep -qiE "power" s5-intel.log && step_ok 5 "power field" || step_fail 5 "power field" "n/a in output (expected)"
else
  step_fail 5 "telemetry tool" "none of nvidia-smi/rocm-smi/intel_gpu_top usable in container"
fi

# return the mounted deliverables to the host user (container runs as root)
if [[ "${HOST_UID:-0}" != "0" ]]; then
  chown -R "${HOST_UID}:${HOST_GID:-0}" "$OUT" 2>/dev/null || true
  chown -R "${HOST_UID}:${HOST_GID:-0}" /hf-cache 2>/dev/null || true
fi

echo "==== spike complete ====" | tee -a summary.txt
EOSPIKE

RC=$?
log "spike exit code: $RC"
log "outputs in $OUT_DIR:"
ls -la "$OUT_DIR" >&2
log "summary:"
cat "$OUT_DIR/summary.txt" >&2 || true
exit $RC