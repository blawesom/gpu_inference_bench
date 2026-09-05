# Intel XPU in Docker: the `/dev/dri` mounting problem

> **Problem**: vLLM's XPU engine-core process crashed at startup inside a Docker
> container with `CCL_ERROR ... init_device_fds: condition dir failed`, even though
> `torch.xpu` could see the GPU.
> **Fix**: bind-mount `/dev/dri` **and** pass `--device` per render node. Both are
> required. This is an Intel XPU + oneCCL + Docker interaction, not a vLLM bug.
>
> Status: resolved 2026-09-04 (commits `e2f0ada`, `96926cd`). Verified on Intel
> Arc Pro B70 (32 GB) with `vllm/vllm-openai-xpu:v0.28.0` under Docker 29.7.2.

## 1. Symptom

The validation probe started the vLLM server and the engine-core process died:

```
CCL_WARN| pidfd is not supported, fallbacks to drmfd exchange mode
CCL_ERROR| ze_fd_manager.cpp:144 init_device_fds: condition dir failed
RuntimeError: Engine core initialization failed. See root cause above.
```

This happened *after* GPU selection had already succeeded (`torch.xpu` reported
one 31.9 GB device), so the failure was not "no GPU visible" — it was inside
oneCCL's inter-process device setup, a component most users never think about.

## 2. What oneCCL does at engine startup

vLLM's XPU engine core shares GPU state across processes via **oneCCL**. oneCCL
must distribute DRM file descriptors to its child processes and has three
exchange modes, tried in this order:

1. **`pidfd`** — the modern Linux path; requires `CAP_SYS_PTRACE`, which plain
   Docker containers **do not have**.
2. **`drmfd`** — fallback: scan the DRM device directory and hand out render-node
   fds.
3. `sockets` — last resort.

Inside Docker, oneCCL therefore always takes mode 2 and logs the benign:

```
CCL_WARN| pidfd is not supported, fallbacks to drmfd exchange mode
```

`drmfd` mode needs to **`opendir()` a directory of render nodes** and match
entries by suffix. From oneCCL source (`src/common/env/env.cpp`):

```cpp
drmfd_dev_render_dir_path("/dev/dri/by-path/"),   // default directory
drmfd_dev_render_suffix("-render"),               // matched suffix
```

So it does `opendir("/dev/dri/by-path/")` and expects entries like
`pci-0000:0b:00.0-render` (symlinks to `/dev/dri/renderD*`). If that directory is
not *listable* inside the container, it dies with
`init_device_fds: condition dir failed` — exactly what we saw.

The override is an env var, `CCL_DRMFD_DEV_RENDER_DIR_PATH` (suffix:
`CCL_DRMFD_DEV_RENDER_SUFFIX`) — useful to know, but we fix the mount instead of
patching env.

## 3. Why each partial fix failed

Docker exposes devices through two **independent** mechanisms:

| Mechanism | What it does |
|---|---|
| `--device /dev/dri/renderD128` | Adds the node to the container's **cgroup device whitelist** — grants `open()` permission; does *not* create a listable `/dev/dri` directory tree |
| `-v /dev/dri:/dev/dri` | **Bind-mounts** the host directory — makes all nodes (and `by-path/`) *visible*; grants **no** cgroup `open()` permission |

Each alone breaks one of the two components:

| Container config | `torch.xpu` (needs `open()` on renderD) | oneCCL (needs listable `/dev/dri/by-path/`) | Result |
|---|---|---|---|
| `--device` per node only | ✅ sees 1 device | ❌ `opendir` fails | engine-core `CCL_ERROR` crash |
| `-v /dev/dri:/dev/dri` only | ❌ `device_count() == 0` (open denied by cgroup) | ✅ directory listable | "no XPU device visible" |
| **both** | ✅ | ✅ | **works** |

Key insight: the bind mount makes the files *visible* but does **not** grant
*permission* to open them — the cgroup v2 device controller still applies, and a
plain container's whitelist does not include arbitrary `renderD*` character
devices. Conversely, `--device` grants permission for the specific node you name
but leaves `/dev/dri` as a non-listable mount with no `by-path/` entries.

## 4. The fix (in `bench.sh`, intel branch)

```bash
[[ -d /dev/dri ]] || die "intel: /dev/dri not found ..."
ls /dev/dri/renderD* >/dev/null 2>&1 || die "intel: no /dev/dri/renderD* nodes ..."
INTEL_NODES=()
for n in /dev/dri/renderD* /dev/dri/card*; do
    [[ -c "$n" ]] && INTEL_NODES+=("$n")
done
GPU_ARGS=(-v /dev/dri:/dev/dri)          # (2) listable dir for oneCCL
for n in "${INTEL_NODES[@]}"; do
    GPU_ARGS+=(--device "$n")            # (1) cgroup open() permission
done
for g in video render; do
    getent group "$g" >/dev/null 2>&1 && GPU_ARGS+=(--group-add "$g")
done
```

Notes:

- All `renderD*` **and** `card*` nodes are passed; on the xe driver the render
  nodes (major:minor 226:128/129) are the compute path, the card nodes are the
  control path. Passing both is harmless and future-proof.
- `--group-add video/render` only added when the group exists (Docker rejects
  unknown group names).
- GPU *selection* is still done **inside** the container by
  `run_matrix.py`/`validate_fit.py` via `ONEAPI_DEVICE_SELECTOR=level_zero:{idx}`
  (colon syntax; verified with a `torch.xpu` probe), because the host may be
  multi-GPU/mixed-vendor and the index is per-vendor, not global.

## 5. Reproducing / verifying the diagnosis

Inside the container (or a scratch container with the same mounts):

```bash
ls -la /dev/dri /dev/dri/by-path/     # by-path must exist and be listable
python3 -c "import torch; print(torch.xpu.device_count())"   # must be >= 1
# then start vLLM; the log may still show the benign CCL_WARN
#   "pidfd is not supported, fallbacks to drmfd exchange mode"
# but must NOT show "init_device_fds: condition dir failed"
```

If you ever see the crash again, the two things to check first are
`ls /dev/dri/by-path` (oneCCL side) and `torch.xpu.device_count()` (open-perm
side) — they isolate which half of the mount is missing.

## 6. Related traps documented along the way

- **`CCL_WARN "pidfd is not supported ..."` is benign** — it is the expected
  fallback to drmfd in Docker. Our log parsers used to false-match "not
  supported" as an *unsupported model* verdict; CCL lines are now filtered out
  before pattern matching (`a9a9254`).
- **`ONEAPI_DEVICE_SELECTOR` is colon syntax** (`level_zero:0`), not
  `level_zero/0` or `level_zero/0:*`.
- **vLLM v1 exits rc=0** when the engine core dies (API server shuts down
  "cleanly"); pre-health exit is treated as CRASH regardless of rc (`a9a9254`).

## 7. References

- oneCCL source: `src/common/env/env.cpp` (defaults + env var names),
  `src/common/global/ze/ze_fd_manager.cpp` (`init_device_fds`)
- vLLM XPU docs (selector format), PyTorch XPU docs (`ONEAPI_DEVICE_SELECTOR`)
- Docker docs: `--device` (cgroup device whitelist semantics), bind mounts vs
  device grants