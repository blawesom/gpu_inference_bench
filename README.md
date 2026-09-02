# gpu_inference_bench

Platform-agnostic GPU inference benchmark: one shell script (`bench.sh`, in
progress) detects the local GPU (NVIDIA / AMD / Intel), pulls the pinned vLLM
image for that vendor, runs the inference stack + a 4-model benchmark matrix
(32–40 GB VRAM fleet), and produces a standalone per-machine report.

- **`PLAN.md`** — the full design (reviewed & final, v3.1): model matrix,
  optimization configs, workload, report schema, script flow, error handling.
- **`spike.sh`** — milestone-2 verification script. Run it on one GPU machine
  (`./spike.sh`) to validate the vLLM v0.28.0 flags / JSON schema / MTP
  speculative-decoding setup before the main implementation.

See `PLAN.md` §12 for the remaining milestones.