# ROCm 10 Validation Error Report

**Date:** 2026-09-13
**GPU:** AMD Radeon AI PRO R9700 (0x7551, 31.9 GB, gfx1201 / RDNA 4)
**Image:** `vllm/vllm-openai-rocm:nightly-rocm100`
**vLLM:** 0.29.1rc1.dev9+g2671fedfc
**PyTorch:** 2.12.0+rocm10.0.0
**HIP:** 7.15.26333
**Host driver:** 7.2.4-1-cachyos
**Kernel:** 7.2.4-1-cachyos
**Docker:** 29.8.0
**OS:** CachyOS (Linux)

## Result

All 5 matrix cells (M1/M3/M4 across baseline, long-context, aiter-attn) **CRASH** during
vLLM engine core initialization. No model can start.

## Root Cause

`torch.AcceleratorError: CUDA error: invalid argument` (`hipErrorInvalidValue`)
during `determine_available_memory()` in the EngineCore init path.

The model weights load fine (M1: 17.65 GiB loaded in 54s), but the very next step —
measuring available GPU memory — triggers a GPU kernel fault. This is a **runtime/driver
incompatibility** between the container's ROCm 10 user-space stack and the host's
ROCm 7.2.4 kernel driver on gfx1201.

## Failing Stack Trace (M1 baseline)

```
(EngineCore) INFO  [core.py:123] Initializing a V1 LLM engine
(EngineCore) INFO  [gpu_worker.py:438] Using V2 Model Runner
(EngineCore) INFO  [model_runner.py:431] Model loading took 17.65 GiB memory and 53.90 seconds
(EngineCore) ERROR [core.py:1366] EngineCore failed to start.
(EngineCore) ERROR [core.py:1366] Traceback (most recent call last):
(EngineCore) ERROR [core.py:1366]   File ".../vllm/v1/engine/core.py", line 1328, in run_engine_core
(EngineCore) ERROR [core.py:1366]   File ".../vllm/v1/engine/core.py", line 1085, in __init__
(EngineCore) ERROR [core.py:1366]   File ".../vllm/v1/engine/core.py", line 145, in __init__
(EngineCore) ERROR [core.py:1366]     kv_cache_config = self._initialize_kv_caches(vllm_config)
(EngineCore) ERROR [core.py:1366]   File ".../vllm/v1/engine/core.py", line 308, in _initialize_kv_caches
(EngineCore) ERROR [core.py:1366]     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore) ERROR [core.py:1366]   File ".../vllm/v1/executor/abstract.py", line 149, in determine_available_memory
(EngineCore) ERROR [core.py:1366]     return self.collective_rpc("determine_available_memory")
(EngineCore) ERROR [core.py:1366]   File ".../vllm/v1/worker/gpu_worker.py", line 569, in determine_available_memory
(EngineCore) ERROR [core.py:1366] torch.AcceleratorError: CUDA error: invalid argument
(EngineCore) ERROR [core.py:1366] Search for `hipErrorInvalidValue' in
(EngineCore) ERROR [core.py:1366]   https://rocm.docs.amd.com/projects/HIP/en/latest/index.html
(EngineCore) ERROR [core.py:1366] CUDA kernel errors might be asynchronously reported at some
(EngineCore) ERROR [core.py:1366]   other API call, so the stacktrace below might be incorrect.
(EngineCore) ERROR [core.py:1366] For debugging consider passing AMD_SERIALIZE_KERNEL=3
(EngineCore) ERROR [core.py:1366] Device-side assertion tracking was not enabled by user.
(APIServer) RuntimeError: Engine core initialization failed. Failed core proc(s): {}
```

## Isolation Tests

| Test | Result |
|---|---|
| `torch.cuda.is_available()` + device name | OK — R9700 detected, 31.9 GB |
| `torch.randn(100, 100, device='cuda')` | OK |
| `a @ a` matmul (1000x1000) | OK |
| `torch.cat` (2 tensors, dim=1) | OK |
| `torch.cat` (list of tensors, dim=1) | OK |
| `F.scaled_dot_product_attention` (contiguous) | OK |
| `F.scaled_dot_product_attention` (10x same tensors) | OK (first 2-3 calls, then `hipErrorInvalidValue`) |
| `F.scaled_dot_product_attention` (split + SDPA + cat, ViT pattern) | **FAIL** on 2nd/3rd chunk |
| Pure text model (Qwen2.5-0.5B-Instruct) `LLM(...)` | **OK** — server started |
| M1 `Qwen3.5-9B` baseline `LLM(...)` | **CRASH** — `hipErrorInvalidValue` |
| M1 `Qwen3.5-9B` long-context `LLM(...)` | **CRASH** — same |
| M1 `Qwen3.5-9B` aiter-attn `LLM(...)` | **CRASH** — same |
| M3 `Qwen3.8-27B-AWQ` baseline `LLM(...)` | **CRASH** — same |
| M4 `Qwen3.5-35B-A3B-AWQ` baseline `LLM(...)` | **CRASH** — same |

**Key insight:** basic GPU ops work, but vLLM's engine core init fails at
`determine_available_memory()`. The issue is NOT ViT-specific (as initially suspected) —
it's a general GPU runtime fault that occurs when vLLM performs its full engine
initialization sequence (memory measurement, KV cache allocation).

## Additional Observations

### 1. ROCProfiler Queue Intercept Warning

Present in every run, from the EngineCore process:

```
W0913 [queue_controller.cpp:170] [queue-intercept] device-memory ring-buffer requested
but profiling requires system-memory InterceptQueue; falling back to system-memory ring
for agent <id> (priority and CU-mask from the descriptor are not preserved)
```

This is a ROCm profiler SDK warning — the device-memory ring buffer is being intercepted
and replaced with a system-memory ring. This may be a contributing factor to the
kernel fault.

### 2. ROCProfiler SDK Library Conflict

A secondary (non-fatal but noisy) issue: two `librocprofiler-sdk.so.1` libraries are
bundled in the image:

```
/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/librocprofiler-sdk.so.1
/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/librocprofiler-sdk.so.1
```

The `ROCPROFILER_REGISTER_LIBRARY` env var is set to the core variant, but the devel
variant tries to override it, causing a check-failure abort in the model registry
inspection subprocess:

```
F0913 registration.cpp:233] ROCPROFILER_REGISTER_LIBRARY is already set to
'.../librocprofiler-sdk.so.1' (resolves to '...'), not overriding with
'.../librocprofiler-sdk.so.1'
*** Check failure stack trace: ***
```

This is a packaging bug in the ROCm 10 nightly image and may be contributing to the
overall instability.

### 3. Image Tag Issue

The `--rocm 10` flag in `bench.sh` constructs the tag
`vllm/vllm-openai-rocm:v0.28.0-rocm10` which **does not exist** on Docker Hub.
Only `nightly-rocm100` (and its commit-pinned variants) carry the `-rocm100` suffix.
Stable version tags (`v0.29.0`, `v0.28.0`, etc.) do not have ROCm 10 variants.

The `bench.sh` tag-construction logic needs updating for when a stable ROCm 10 image
is released, and the nightly tag must be explicitly specified via `--image` for now.

### 4. Host/Container Driver Mismatch

| Component | Version |
|---|---|
| Host kernel driver (amdgpu) | 7.2.4 |
| Container ROCm user-space | 10.0.0 (HIP 7.15) |
| Container PyTorch | 2.12.0+rocm10.0.0 |
| Container vLLM | 0.29.1rc1.dev9 |

The container's ROCm 10 user-space stack runs against the host's ROCm 7.2.4 kernel
driver. This mismatch is a likely root cause — the ROCm 10 user-space may be issuing
HIP kernel launches that the 7.2.4 driver doesn't understand.

## AITER Performance Investigation

Separate from the ROCm 10 nightly crash above, the **stable** image
(`vllm/vllm-openai-rocm:v0.28.0`, ROCm 7.2.53211, run
`results/20260913-141852_0x7551`) was used to evaluate the T1 AITER
unified-attention cell. **AITER provides no performance improvement because it
never reaches inference** — the `M1 / aiter-attn` cell fails at engine startup
with `pass1-failed:engine-startup` and no throughput is captured.

### Failure mode: GPU Hang during Triton autotuning

The AITER server log (`server_M1_aiter-attn_pass1.log`) shows a clean startup up
to the profiling stage, then a **hardware-level GPU fault**:

1. `module_aiter_core.so` imports successfully (line 11).
2. Backend selection works — `ROCM_AITER_UNIFIED_ATTN` is chosen over
   `ROCM_ATTN` / `TRITON_ATTN` (line 34), and
   `RocmAiterUnifiedAttentionImpl` is confirmed (line 35).
3. KV cache block size is set to 64 for the AITER backend (line 48).
4. During the initial profiling/warmup run, **Triton autotuning of
   `chunk_fwd_kernel_o`** (a GDN linear-attention kernel) crashes the GPU:

   ```
   Triton autotuning for function chunk_fwd_kernel_o, finished after 11.20s,
   best config selected: BK: 64, BV: 32, num_warps: 4, num_ctas: 1, num_stages: 2
   INFO [monitor.py:81] Initial profiling/warmup run took 38.86 s
   Memory access fault by GPU node-2 ... on address 0x7f400a90d000.
       Reason: Page not present or supervisor privilege.
   HW Exception by GPU node-2 ... reason :GPU Hang
   GPU coredump: execvp failed: No such file or directory
   GPU core dump failed
   ```

   The process is killed before any benchmark pass can start, so
   `concurrency_results` is empty and the cell is marked `failed`.

### Baseline comparison (same run, same GPU)

The `M1 / baseline` cell runs fine on the same GPU. Its attention path is
**different** from AITER's:

- `backend_override`: `ROCM_ATTN` (Triton fallback) —
  `Cannot use ROCm custom paged attention kernel, falling back to Triton
  implementation`.
- No GPU fault; throughput captured (277.8 tok/s @ C=16, 1296 J/ktok).

So the "no improvement from AITER" observation is not a modest kernel win — it
is a **hard crash**. The AITER unified-attention path is not yet stable on
RDNA4 (gfx1201) under vLLM 0.28.0 + ROCm 7.2.5.

### Suspected root causes

1. **AITER unified attention is immature on RDNA4** — gfx1201 is a new
   architecture and the AITER kernels (and the surrounding GDN linear-attention
   Triton autotuning they pull in) are not yet validated for it.
2. **The crash is in autotuning, not steady-state inference** — the fault hits
   during the one-off kernel-config sweep at startup, so even a correct steady
   state would never be observed.
3. **M1 is a hybrid Mamba/GDN model** — the GDN linear-attention kernels that
   autotune (and fault) are part of M1's architecture, not the AITER dense
   attention itself. The AITER backend only claims the dense-attention layers;
   the GDN path is unchanged and is where the hang originates.

### Relationship to the ROCm 10 crash

The two failures are **independent but both block AITER on this GPU**:

- **ROCm 7.2.5 / vLLM 0.28.0** (stable): engine starts, AITER backend is
  selected, but a **GPU hang** kills it during autotuning.
- **ROCm 10 nightly / vLLM 0.29.1rc1**: the **host/driver mismatch**
  (`hipErrorInvalidValue`) prevents *any* cell (including baseline) from
  starting at all — see the sections above.

## Fit Validation Summary

| Model | Config | Verdict | Weights GB | Util | KV/token |
|---|---|---|---|---|---|
| M1 (Qwen3.5-9B) | baseline | **CRASH** | 19.3 | 0.9 | 32.0KB |
| M1 (Qwen3.5-9B) | long-context | **CRASH** | 19.3 | 0.9 | 32.0KB |
| M1 (Qwen3.5-9B) | aiter-attn | **CRASH** | 19.3 | 0.9 | 32.0KB |
| M3 (Qwen3.8-27B-AWQ) | baseline | **CRASH** | 21.0 | 0.95 | 64.0KB |
| M4 (Qwen3.5-35B-A3B-AWQ) | baseline | **CRASH** | 24.5 | 0.95 | 20.0KB |

**Result: 0 FIT, 0 TIGHT, 5 CRASH of 5 cells**

Full fit report:
`results/validate_20260913-214634_card-series-amd-radeon-ai-pro-r9700/fit_report.md`
Probe logs: `results/validate_20260913-214634_card-series-amd-radeon-ai-pro-r9700/probe_*.log`

## Next Steps

1. **Try stable `v0.29.0` image** — pull `vllm/vllm-openai-rocm:v0.29.0` and check if it
   bundles ROCm 10 or 7.x. If it's ROCm 10, re-run validation.
2. **Try other nightly variants** — older `nightly-rocm100-<commit>` tags may predate
   the bug:
   - `nightly-rocm100-385dce36bcee42309924a5ece951a96db3dce7f2`
   - `nightly-rocm100-9ea8f3ffc354901b740f0b31988900897b7221d7`
   - `nightly-rocm100-d9105ea8001e0a6d77a96327d17515bb5791fb36`
3. **Update host driver** — if a ROCm 10 kernel driver is available for CachyOS,
   installing it would eliminate the driver/user-space version mismatch.
4. **File upstream issue** — report `hipErrorInvalidValue` on gfx1201 with
    ROCm 10.0 + PyTorch 2.12.0 to the vLLM/ROCm teams.
 5. **AITER on ROCm 7.2.5 (stable)** — the AITER GPU-hang is independent of the
    ROCm 10 crash. To actually measure AITER:
    - Confirm the hang is AITER-specific: run M1 baseline with
      `VLLM_ROCM_USE_AITER=1` but **without** `ROCM_AITER_UNIFIED_ATTN=1` (i.e.
      let AITER load but keep `ROCM_ATTN`) to see if the GDN autotuning fault is
      triggered by the AITER backend or by AITER import alone.
    - Check whether a newer AITER build (newer vLLM / ROCm 7.2.x point release)
      ships a fixed `chunk_fwd_kernel_o` / GDN path for gfx1201.
    - File the GPU-hang (`Memory access fault` → `GPU Hang` during
      `chunk_fwd_kernel_o` autotuning) against the AITER / vLLM-ROCm RDNA4
      tracker, with `server_M1_aiter-attn_pass1.log` as evidence.

## Reproduction

```bash
# Pull the image
docker pull vllm/vllm-openai-rocm:nightly-rocm100

# Run the validation
./bench.sh --validate --vendor amd --image vllm/vllm-openai-rocm:nightly-rocm100
```