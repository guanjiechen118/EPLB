## models
1. deepseekV2-lite : /mnt/shared-storage-user/moegroup/share_models/DeepSeek-V2-Lite
2. deepseekV3 : /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/afb92e1fa402c2be2a9eb085312bb02e0384d6c7
3. Qwen3.5: 
    - 397B-A17B: /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/285b7b5d3792e7357b31101b858806a0eddd3e3c/
    - 122B-A10B: /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99/
4. Qwen3: /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/4c446470ba0aec43e22ac1128f9ffd915f338ba3/

## fgate-only / fgate-hybrid-cache

`vllm_eplb` now supports EPLB `fgate-only` and `fgate-hybrid-cache` for:

- DeepSeek-V2-Lite (`DeepseekV2ForCausalLM`)
- Qwen3-30B-A3B (`Qwen3MoeForCausalLM`)
- Qwen3.5-122B-A10B / 397B-A17B (`Qwen3_5MoeForConditionalGeneration`, with the language model routed by `Qwen3NextSparseMoeBlock`)

### Algorithm summary

- `fgate-only`: correct idea2-style local-shadow immediate refresh. All redundant slots are split into two double-buffered local shadow banks. After layer `l` predicts layer `l+1` demand, the runtime immediately stages layer `l+1`'s standby bank and flips it at the next layer boundary. This path is decode-focused, local-only, and never mutates the global runtime routing map.
- `fgate-hybrid-cache`: hybrid same-node peer cache. A static subset of redundant slots is managed by EMA, while the remaining redundant slots are split into two double-buffered fgate banks so the runtime can prefetch the next predicted expert set before swapping banks.

Example `eplb-config` values:

```bash
--enable-eplb --eplb-config '{"algorithm":"fgate-only","window_size":1000,"step_interval":1000,"num_redundant_experts":8,"log_balancedness":true}'

--enable-eplb --eplb-config '{"algorithm":"fgate-hybrid-cache","window_size":1000,"step_interval":1000,"num_redundant_experts":16,"num_static_redundant_experts":8,"log_balancedness":true}'
```

## scripts

A compact script directory is available at:

`/mnt/shared-storage-user/chenguanjie/huawei_eplb/scripts`

Files:

- `scripts/serve_eplb.sh`: start a server with model profiles and `fgate` / `fgate-v2`
  该脚本里的旧算法名如果仍存在，需要改成 `fgate-only`。
- `scripts/bench_serve.sh`: run `vllm bench serve` with either `random` or `custom` datasets
- `scripts/common.sh`: model path profiles and default max length values

### MODEL_KEY options

- `deepseek_v2_lite`
- `deepseek_v3`
- `qwen3_30b_a3b`
- `qwen3_5_122b_a10b`
- `qwen3_5_397b_a17b`

### Examples

Edit the variables at the top of the script first, then run:

```bash
cd /mnt/shared-storage-user/chenguanjie/huawei_eplb

bash scripts/serve_eplb.sh
bash scripts/bench_serve.sh
```
