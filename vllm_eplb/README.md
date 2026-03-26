# vllm_v13 Local Change Log

This README replaces the upstream vLLM README and records the local changes made on this development node.

- Repo: `/mnt/shared-storage-user/chenguanjie/vllm_v13`
- Date: `2026-03-19`
- Formal branch: `main`
- Source feature branch: `feat/eplb-fgate`
- Key feature commits:
  - `b3239944b` - `Add fgate algorithm support to EPLB`
  - `4dc9a89fe` - `Extend fgate support to Qwen3 MoE`

---

## Goal

Add `fgate` support into EPLB for the editable local vLLM tree, following the reference changes in:

- reference repo: `/mnt/shared-storage-user/chenguanjie/vllm`

Also add a small runtime stability patch so EPLB rearrangement works under the current DP + MLA serving setup.

---

## Summary of local changes

### 1. Added EPLB algorithm selection

Added support for the following EPLB load-estimation algorithms:

- `swm`
- `ema`
- `fgate`

New EPLB config fields:

- `eplb_config.algorithm`
- `eplb_config.ema_alpha`

Example:

```bash
--enable-eplb \
--eplb-config '{"window_size":1000,"step_interval":10,"num_redundant_experts":2,"log_balancedness":true,"algorithm":"fgate"}'
```

---

### 2. Added fgate logic into EPLB state

Implemented `fgate` handling in EPLB state management:

- allocate predicted-load buffer for fgate
- allocate fgate sliding window
- update predicted load every step
- use fgate predicted load during expert rearrangement

Relevant file:

- `vllm/distributed/eplb/eplb_state.py`

---

### 3. Added DeepSeek-V2 / DeepSeek-V3 fgate predicted-load path

Implemented next-layer gate based predicted-load accumulation for DeepSeek-V2 MoE.

DeepSeek-V3 is covered by the same implementation/class hierarchy in the same file.

Idea:

- current layer hidden states are projected by the next layer's gate weight
- predicted gate scores are accumulated into EPLB's fgate load tensor

Relevant file:

- `vllm/model_executor/models/deepseek_v2.py`

---

### 4. Added Qwen3MoE / Qwen3-VL-MoE fgate predicted-load path

Implemented the same fgate mechanism for Qwen3 MoE-family routed layers.

Idea:

- current layer hidden states are projected by the next layer's gate weight
- predicted gate scores are accumulated into EPLB's fgate load tensor

Relevant files:

- `vllm/model_executor/models/qwen3_moe.py`
- `vllm/model_executor/models/qwen3_vl_moe.py`

---

### 5. Extended MoE EPLB state plumbing

Added extra EPLB state wiring so MoE layers can receive:

- `next_gate_weight`
- `expert_load_fgate_view`

Relevant files:

- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/models/interfaces.py`
- `vllm/model_executor/models/afmoe.py`
- `vllm/model_executor/models/transformers/moe.py`
- `vllm/model_executor/models/mllama4.py`

---

### 6. Added current EPLB algorithm log

Added logging during EPLB initialization so startup logs show the actual algorithm in use.

Expected log format:

```text
EPLB initialized for model <model_name>: algorithm=fgate, policy=default, window_size=..., step_interval=..., ema_alpha=...
```

This helps confirm the model is truly using `fgate`.

Relevant file:

- `vllm/distributed/eplb/eplb_state.py`

---

## Stability patch added after testing

During validation, the server could crash when EPLB performed its first rearrangement under:

- `data_parallel_size=2`
- MLA attention backend
- v1 engine
- DP dummy-batch synchronization path

Observed failure:

- benchmark requests succeed
- server crashes around/after first EPLB rearrangement
- stack ends in:

```text
vllm/v1/attention/backends/utils.py
AssertionError: tokens not padded correctly
```

### Root cause (practical view)

When DP ranks become temporarily misaligned around EPLB rearrangement, vLLM may execute a `dummy batch` to keep collectives aligned.
That dummy batch was entering a fragile cudagraph-based MLA metadata path.

### Minimal fix applied

Only one low-level runtime patch was kept:

File:

- `vllm/v1/worker/gpu_worker.py`

Change:

```python
self.model_runner._dummy_run(1, uniform_decode=True)
```

to:

```python
self.model_runner._dummy_run(
    1,
    uniform_decode=True,
    cudagraph_runtime_mode=CUDAGraphMode.NONE,
)
```

### Effect of this patch

- only affects DP synchronization dummy batches
- does **not** change normal request execution path
- does **not** disable EPLB rearrangement
- does **not** remove `fgate`
- avoids the crashing dummy-batch cudagraph/MLA path

---

## Files changed

### fgate / EPLB feature files

- `vllm/config/parallel.py`
- `vllm/distributed/eplb/eplb_state.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/models/deepseek_v2.py`
- `vllm/model_executor/models/qwen3_moe.py`
- `vllm/model_executor/models/qwen3_vl_moe.py`
- `vllm/model_executor/models/interfaces.py`
- `vllm/model_executor/models/afmoe.py`
- `vllm/model_executor/models/transformers/moe.py`
- `vllm/model_executor/models/mllama4.py`

### runtime stability patch

- `vllm/v1/worker/gpu_worker.py`

---

## setup.py local changes present in this repo

Current local `setup.py` includes the following repo-specific changes:

1. A debugging breakpoint was added in the nightly wheel metadata fetch path:

```python
print(f"Trying to fetch nightly build metadata from {meta_url}")
breakpoint()
```

2. FA3 extension registration was commented out:

```python
ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
```

Practical effect:

- local builds may drop into debugger while resolving precompiled wheel metadata
- FA3 extension is not registered from `setup.py` in the current tree

These setup changes are local to this repository state and should be reviewed before production packaging or upstreaming.

---

## Reproduction notes from testing

### Server start script used

Path:

- `/mnt/shared-storage-user/chenguanjie/scripts/start_server.sh`

Important options:

- `--data-parallel-size 2`
- `--enable-expert-parallel`
- `--enable-eplb`
- `--eplb-config '{..."algorithm":"fgate"...}'`

### Benchmark script used

Path:

- `/mnt/shared-storage-user/chenguanjie/scripts/run_benchmark.sh`

### Observed behavior

- benchmark itself can finish successfully
- server crash appears when EPLB reaches rearrangement and DP executes dummy-batch synchronization

---

## Notes for future debugging

1. `step_interval` is not perceived as a simple “first rearrange after N requests” counter in practice.
   In real serving, EPLB step progression depends on model runner execution steps, not benchmark request count directly.

2. If crashes reappear around rearrangement, check:
   - DP dummy-batch path
   - MLA backend
   - whether `execute_dummy_batch()` changed again

3. To confirm current branch:

```bash
git branch --show-current
```

4. To confirm latest local commit:

```bash
git log --oneline -5
```

---

## Current expected branch state

Formal branch:

```bash
main
```

Source feature branch:

```bash
feat/eplb-fgate
```

Feature commits:

```bash
b3239944b Add fgate algorithm support to EPLB
4dc9a89fe Extend fgate support to Qwen3 MoE
```

---

## Quick reminder

This repository is no longer using the upstream generic README.
This file is intentionally local and task-oriented, meant to record the modifications made for:

- EPLB `fgate`
- DeepSeek-V2 next-gate predicted load
- DP dummy-batch stability around EPLB rearrangement
