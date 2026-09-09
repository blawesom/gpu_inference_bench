#!/usr/bin/env bash
# Decisive check for Intel GPU power telemetry (see docs/intel-telemetry-eval.md):
# does xpu-smi work INSIDE the stock vLLM XPU image, with the same DRI
# passthrough that bench.sh uses for real runs?
#
# Usage (on the Intel host, from anywhere):
#   ./container/test_xpu_smi.sh                    # defaults to vllm/vllm-openai-xpu:v0.28.0
#   ./container/test_xpu_smi.sh <image>            # any other vLLM XPU image
#
# Exit 0 with a wattage from `dump -m power`  -> stock image is sufficient,
#   no wrapper image / --build-xpu-tools needed; telemetry picks xpu-smi up
#   automatically.
# "Sysman driver initialization failed" even with devices -> environment/
#   permission gap to chase (compare with the host, where xpu-smi works).

set -u

IMAGE="${1:-vllm/vllm-openai-xpu:v0.28.0}"

if [[ ! -d /dev/dri ]]; then
    echo "ERROR: /dev/dri not found — is the xe kernel module loaded? (lsmod | grep xe)" >&2
    exit 1
fi

NODES=()
for n in /dev/dri/renderD* /dev/dri/card*; do
    [[ -c "$n" ]] && NODES+=("$n")
done
if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "ERROR: no /dev/dri/renderD* or card* nodes found" >&2
    exit 1
fi

DEVICES=()
for n in "${NODES[@]}"; do DEVICES+=(--device "$n"); done

GROUPS=()
GNAMES=()
for g in video render; do
    if getent group "$g" >/dev/null 2>&1; then
        GROUPS+=(--group-add "$g"); GNAMES+=("$g")
    fi
done

echo "image:   $IMAGE"
echo "devices: ${NODES[*]}"
echo "groups:  ${GNAMES[*]:-none}"
echo

docker run --rm \
    --entrypoint bash \
    -v /dev/dri:/dev/dri \
    "${DEVICES[@]}" \
    ${GROUPS[@]+"${GROUPS[@]}"} \
    "$IMAGE" \
    -c '
        echo "=== xpu-smi on PATH ==="; command -v xpu-smi || echo MISSING
        echo
        echo "=== version ==="; xpu-smi --version 2>&1 | head -3
        echo
        echo "=== discovery ==="; xpu-smi discovery 2>&1 | head -8
        echo
        echo "=== dump -m power (decisive) ==="
        xpu-smi dump -d 0 -m power 2>&1
    '