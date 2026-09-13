# Spec — P1 power-window alignment · T1 AITER attention

Status: **implemented** (2026-09-09) — built as specified; first GPU
validation pending the T0/T1 campaign.

Scope: two items from the 2026-09-08 review
(`docs/Benchmark_GPU_Conclusions_et_plan_de_tests_Benjamin.pdf`):

1. **P1** (section 01, last item): align power and throughput to the same
   window, integrate joules, separate GPU energy from system energy.
2. **T1** (section 03, plan row T1 / section 02 "Attention"): compare the
   current ROCm attention path (which falls back to Triton) against the
   AITER unified-attention path. AMD only, M1 first.

Both inherit the P0 protocol (3 passes, per-level warmup, token guard,
prefix cache verified) — nothing here special-cases measurement.

---

## Item 1 · P1 power-window alignment

### 1.1 Current behavior (verified)

- `run_matrix.py` starts the 1 Hz sampler **before** `vllm bench serve` and
  stops it **after** the client exits. Sampled window = client startup
  (python import, dataset build, connect) + in-bench warmups (`--num-warmups
  2`) + the **measured window** + client teardown.
- `vllm bench serve` reports `duration` = the **measured window only**
  (e.g. `bench_M2_baseline_16.json`: `duration` 12.8 s vs. client wall
  ~15 s). Throughput is computed over that window; `power_avg_w` is computed
  over the wider window. **The two windows are misaligned.**
- `container/telemetry.py` samples carry a timestamp (`sample["t"]`,
  epoch) — window alignment is possible from existing data shapes.
- The telemetry JSON stores **only the aggregate** (e.g.
  `telemetry_M3_baseline_16.json` keys: `n_samples, duration_s,
  mem_avg/peak, util_avg/peak, power_avg/peak`). Raw samples are dropped,
  so **existing runs cannot be retro-aligned**.
- `docs/compare_runs.py` uses `power_avg_w` directly for tok/s/W; Intel has
  no power telemetry at all and is evaluated at an assumed 230 W TDP (the
  note: "un TDP n'est pas une mesure" — a TDP is not a measurement).

### 1.2 What the note requires

> "Aligner puissance et débit sur la même fenêtre et intégrer les joules.
> Distinguer énergie GPU et énergie totale du système ; sinon ne pas
> publier de classement énergétique définitif."

### 1.3 Design

**telemetry.py**

- `aggregate(samples, label=None)` — add, when `power_w` samples exist:
  - `energy_j`: time-integrated energy `Σ P_i · Δt_i` using **actual**
    inter-sample intervals (not assumed 1.0 s); null if < 2 power samples.
  - Keep all existing fields unchanged.
- New module function `align_window(samples, wall_s, window_s)`:
  - Returns the sub-list of samples with `t` in
    `[t_first + (wall_s − window_s), t_first + wall_s]` (times relative to
    sampler start ≈ client start).
  - Rationale: the measured window is the **last `duration` seconds** of
    the client wall time. Client teardown (percentile math, JSON save) is
    < ~2 s and lands after the window, so anchoring on the process end is
    the tightest anchor available from artifacts we already have.
  - Degenerate cases: `window_s >= wall_s` or < 2 samples inside the window
    → return `None` (caller keeps the full window and records the reason).

**run_matrix.py** (per measured bench, per pass)

- After the bench subprocess: read `duration` and
  `total_output_tokens` from the bench JSON.
- Aligned samples = `align_window(samples, wall, duration)`.
- Telemetry JSON (per pass) becomes:

  ```jsonc
  {
    "window": {
      "aligned": true,                  // false when align_window returned None
      "reason": null,                   // or "no-duration" | "insufficient-samples"
      "wall_s": 15.2,
      "measured_window_s": 12.8,        // bench JSON `duration`
      "n_samples": 13,                  // samples inside the aligned window
      "n_samples_full": 15
    },
    "energy_j": 3421.5,                 // aligned window, GPU-only
    "energy_j_per_ktok": 155.9,         // energy_j / (total_output_tokens/1000)
    "power_avg_w": 263.0,               // ALIGNED window (primary value)
    "power_peak_w": 301.0,
    "mem_peak_gb": 30.1,                // ... other aggregates over aligned window
    "util_avg_pct": 97.2,
    "full_window": {                    // traceability: pre-alignment values
      "power_avg_w": 254.3,
      "power_peak_w": 301.0,
      "energy_j": 3890.2
    },
    "samples": [ {"t": 0.4, "power_w": 12.0, "util_pct": 0,
                  "mem_used_gb": 21.3}, ... ]   // raw 1 Hz samples
  }
  ```

- `power_avg_w` stays the **aligned** value so `report.py` and
  `compare_runs.py` pick it up with no logic change.
- `energy_j_per_ktok` is null when `total_output_tokens` is 0/missing or
  the level is token-shortfall-flagged (a short run's energy/token is
  meaningless; the flag already marks the level).

**Semantics & labeling (the note's third requirement)**

- All energy values are **GPU energy** from the vendor GPU power sensor
  (`nvidia-smi power.draw`, `rocm-smi` power column). System-level energy
  (CPU, memory, PSU losses) is **not** measured and must never be implied.
- `report.md`: the telemetry table gains one column **J/ktok**; the Data
  quality section gains one line per run: "energy = GPU-only (vendor power
  sensor); window aligned to the measured bench window" — or, for legacy
  runs (no `window` key): "power/energy over the **full bench client
  window** (pre-P1); energy/token is an upper bound, avg power a lower
  bound".
- `compare_runs.py` power section: footnote distinguishes aligned vs
  legacy windows per run (telemetry `window` key), and repeats that Intel
  remains TDP-assumed until power telemetry exists (prerequisite:
  `docs/intel-telemetry-eval.md`). **No definitive energy ranking is
  published while Intel is TDP-assumed** (per the note) — the table keeps
  the existing `*` marker and the wording "conservative floors".

**Direction of impact on existing numbers (for the record)**

- The client-startup tail runs at low GPU power, so today's `power_avg_w`
  under-states steady-state power → today's tok/s/W is **optimistic** for
  all three systems; today's energy/token is pessimistic. Alignment moves
  efficiency numbers **down** and J/ktok **up**. The cross-system ranking
  could in principle change; that is the point of the fix.

**Legacy runs (Sept 3–5)**

- No raw samples → not re-alignable. Reports keep current values + the
  "full client window" data-quality line. Nothing is recomputed or
  relabeled as aligned.

### 1.4 Validation

- Synthetic: unit-style check of `align_window` (window shorter than
  span; window ≥ span; single-sample window; samples with/without power)
  and `energy_j` (constant 100 W over 10 s → 1000 J ± interval noise).
- Dry run: telemetry file shape unchanged in name
  (`telemetry_<cell>_<C>_pass<N>.json`); `--dry-run` output unchanged.
- First GPU run: for one cell, verify by eye that
  `power_avg_w (aligned) ≥ power_avg_w (full_window)` and that
  `n_samples ≈ duration ± 2`.

### 1.5 Out of scope

- System-level energy (RAPL/PSU) — not available on any of the three
  hosts today; would need host-side instrumentation (separate spec).
- Intel power telemetry acquisition itself — tracked in
  `docs/intel-telemetry-eval.md`; this item only consumes it when present.

---

## Item 2 · T1 AITER attention

### 2.1 Current behavior (verified)

- ROCm paged attention **falls back to Triton** on the 2026-09-03 AMD run:
  `server_M3_baseline.log` L68: `Cannot use ROCm custom paged attention
  kernel, falling back to Triton implementation.` Backend selection is
  logged at L34: `Found incompatible backend(s) [TURBOQUANT] with
  AttentionType.DECODER. Overriding with ROCM_ATTN out of potential
  backends: ['ROCM_ATTN', 'TRITON_ATTN'].`
- The vLLM 0.28.0 ROCm image references AITER ops
  (`rocm_aiter_sparse_attn_indexer` appears in the engine compilation
  config) — the library is at least importable; the **version and its
  compatibility with this vLLM build are unverified**.
- The orchestrator has **no per-cell environment** today: the server
  inherits the container env; there is no `env:` layer alongside
  `flags:`.

### 2.2 What the note requires (T1 row, section 03 + "Attention", section 02)

- Compare the current backend to `ROCM_AITER_UNIFIED_ATTN` with
  `VLLM_ROCM_USE_AITER=1` and a **compatible AITER version installed**.
- AMD only; **M1 first**; extend to M3/M4 only if compatible.
- Same model, same parameters — **only the attention path changes**
  (one-variable-at-a-time; revert to the T0 reference before changing
  anything else).
- Verify: effective backend selection in the logs; numerical validity;
  reproducible throughput/latency improvement; quality verified.
- Explicit warning: RDNA4 AITER kernels differ from Instinct CK/ASM —
  **do not copy an MI300 profile**.

### 2.3 Design

**New config `aiter-attn` in `config/models.yaml`**

```yaml
configs:
  aiter-attn:
    # T1 (2026-09-08 review): ROCm paged attention falls back to Triton on
    # this stack (see server logs). Try the AITER unified-attention path.
    # ONLY the attention path changes vs baseline — same model, flags,
    # workload. AMD only, M1 first; extend to M3/M4 only after M1 shows a
    # clean AITER selection in the logs and the quality check passes.
    flags: {}
    env:
      VLLM_ROCM_USE_AITER: "1"
      ROCM_AITER_UNIFIED_ATTN: "1"
```

- Added to **M1's** `configs:` list only (`[baseline, long-context,
  aiter-attn]`). M3/M4 unchanged until the compat gate below passes.
- **Non-AMD runs**: `run_cell` auto-skips this config with
  `skipped:aiter-amd-only` (explicit in the report, exit code unaffected —
  a skip is not a failure, same as existing `skipped:` reasons).

**Per-cell environment support (new, run_matrix.py)**

- `env:` maps accepted in the same three layers as `flags:`
  (`common_server` < model < config, later overrides earlier, string
  values only).
- Merged env is passed to the server process only
  (`start_server(cmd, log_path, env=None)` → `Popen(..., env=env)`); the
  bench client and warmups are unaffected.
- Traceability (section 04): the effective env **delta** vs the container
  env is recorded in the cell manifest (`cell["server_env"]`), and
  `environment.json` gains `stack.aiter` (best-effort
  `python3 -c "import aiter; print(getattr(aiter, '__version__',
  'unknown'))"` on AMD runs) so every trial archives the AITER version, as
  the note's deliverables spec requires.

**Effective-selection verification (note: "vérifier la sélection
effective")**

- After the health check (alongside the P0-1 prefix-cache verification),
  scan the server log and record in
  `cell["server_flags"]["attention_evidence"]`:
  - `aiter_lines`: first ≤ 3 lines matching `aiter` (case-insensitive),
    each truncated to 200 chars — evidence the path is engaged;
  - `triton_fallback`: first line matching `falling back to Triton`, or
    null;
  - `backend_override`: first line matching `Overriding with .* out of
    potential backends`, or null (the L34-style selection dump).
- Data quality (report.py): config `aiter-attn` with **no** `aiter_lines`
  → issue `aiter-unverified` ("AITER env set but no AITER evidence in
  server log — backend may have silently fallen back; do not compare
  against baseline"). Config `baseline` with `triton_fallback` →
  informational entry only (expected on this stack; it is the baseline
  state, not a defect).

**Numerical-validity capture (note: "validité numérique")**

- For AMD cells with config in `{baseline, aiter-attn}`, right after
  startup verification (before the sweep), send **3 fixed prompts**
  (deterministic constant in the orchestrator, temperature 0,
  `max_tokens=64`, `ignore_eos` off) to `/v1/chat/completions` and save the
  raw responses to `diag_aiter_<cell>.json`.
- Post-run check (manual, documented in README): diff baseline vs
  aiter-attn token-for-token. Identical outputs = strong agreement. A
  handful of late-token flips = plausible float-order tolerance. Systematic
  divergence = stop, do not compare. The bench does **not** auto-gate on
  this (it is a diagnostic artifact, like the P0-2 `diag_*.json` files).

**Quality gate (manual)**

- Per the note's T1 validation criteria and the "Qualité" deliverable:
  before any AITER number is accepted, run the fixed representative corpus
  with the same scoring protocol on baseline and aiter-attn. This is a
  documented manual step (README, T1 checklist) — no scoring harness is
  added to the repo by this spec.

### 2.4 Validation criteria (from the note, restated for the run)

| Criterion | How it is met |
|---|---|
| Backend confirmed in logs | `attention_evidence.aiter_lines` non-empty; else cell flagged `aiter-unverified` |
| Reproducible improvement | 3-pass median + spread (P0 protocol); improvement must exceed pass dispersion |
| Numerical validity | `diag_aiter_*.json` baseline-vs-aiter diff (manual) |
| Quality verified | fixed corpus, same scoring (manual gate) |

### 2.5 Risks & open questions

1. **AITER version compatibility** with vLLM 0.28.0+rocm723 is unknown.
   Mitigation: the evidence scan + `stack.aiter` version capture make any
   silent fallback or bad pairing visible; server start failure is already
   surfaced as `failed:engine-startup` with the log archived.
   *Open: should we pin an AITER version in the image, or accept the
   image-shipped one for the first trial?* (Spec default: image-shipped,
   version recorded; a pinned version is a follow-up if the first trial is
   inconclusive.)
2. **M1 is a hybrid Mamba/GDN model** (`Qwen3.5-9B`): AITER unified
   attention should only touch the attention part; the Mamba/GDN paths are
   unchanged. The log evidence will show what was actually selected.
3. **M3/M4 extension** is deliberately deferred (compat gate = M1 clean
   selection + numerical check + quality).
4. No MI300/Instinct profile or kernel tuning is copied anywhere — the
   trial is env-vars-only, per the note's warning.

### 2.6 Out of scope

- AITER-based MoE/linear kernels (that is T2 territory — M4 MoE tuning).
- Stack upgrades / validated R9700 stack (T5).
- KV-FP8 interaction with AITER (T4, after T0).

---

## Execution order

1. **P1 power alignment** — pure repo work; no GPU needed to validate the
   logic (synthetic + dry run). Ships independently of AITER.
2. **T1 AITER** — repo work (config, env support, evidence scan, diag
   capture) + one **AMD GPU** run of M1 `{baseline, aiter-attn}` under the
   T0 protocol. The run itself belongs to the post-T0 campaign: the first
   AITER numbers are only meaningful against a corrected (P0/P1) baseline.

## Files touched (planned)

| File | Item 1 (P1) | Item 2 (T1) |
|---|---|---|
| `container/telemetry.py` | `energy_j`, `align_window()` | — |
| `container/run_matrix.py` | aligned telemetry JSON, raw samples, J/ktok | `env:` layers, `start_server(env=)`, auto-skip non-AMD, attention evidence, `diag_aiter_*.json`, `stack.aiter` |
| `container/report.py` | J/ktok column, window lines in Data quality | `aiter-unverified` issue, attention evidence in Data quality |
| `config/models.yaml` | — | `aiter-attn` config, M1 list, env-layer doc |
| `docs/compare_runs.py` | window footnote, energy wording | — |
| `README.md` | energy semantics | T1 checklist (manual gates) |