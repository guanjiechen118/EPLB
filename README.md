## models
1. deepseekV2-lite : /mnt/shared-storage-user/moegroup/share_models/DeepSeek-V2-Lite
2. deepseekV3 : /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3-Base/snapshots/afb92e1fa402c2be2a9eb085312bb02e0384d6c7
3. Qwen3.5: 
    - 397B-A17B: /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/285b7b5d3792e7357b31101b858806a0eddd3e3c/
    - 122B-A10B: /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99/
4. Qwen3: /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/4c446470ba0aec43e22ac1128f9ffd915f338ba3/

## fgate / fgate-v2

`vllm_eplb` now supports EPLB `fgate` and `fgate-v2` for:

- DeepSeek-V2-Lite (`DeepseekV2ForCausalLM`)
- Qwen3-30B-A3B (`Qwen3MoeForCausalLM`)
- Qwen3.5-122B-A10B / 397B-A17B (`Qwen3_5MoeForConditionalGeneration`, with the language model routed by `Qwen3NextSparseMoeBlock`)

### Algorithm summary

- `fgate`: after the current layer finishes routing, use the current hidden states and the **next** MoE gate to predict the next-layer expert load, then feed that prediction into EPLB.
- `fgate-v2`: same as `fgate`, but skips the prediction during prefill (`max_query_len > 1`) and only accumulates predicted load during decode. This reduces prefill overhead for long-context serving.

Example `eplb-config` values:

```bash
--enable-eplb --eplb-config '{"algorithm":"fgate","window_size":1000,"step_interval":1000,"num_redundant_experts":2,"log_balancedness":true}'

--enable-eplb --eplb-config '{"algorithm":"fgate-v2","window_size":1000,"step_interval":1000,"num_redundant_experts":2,"log_balancedness":true}'
```

## scripts

A compact script directory is available at:

`/mnt/shared-storage-user/chenguanjie/huawei_eplb/scripts`

Files:

- `scripts/serve_eplb.sh`: start a server with model profiles and `fgate` / `fgate-v2`
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
