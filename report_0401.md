# Report 0401

## 1. Commit 信息

- Commit: `745238ef04711bac55232eeb1d81ac1005e44215`
- Short hash: `745238e`
- Message: `support hybrid fgate stratefy`
- 时间: `2026-04-01 05:52:34 +0000`

本次提交是一次较大的阶段性整合，覆盖了 hybrid fgate 核心实现、周期性重排稳定性修复、实验脚本重构、domain-shift 数据集构建、复现实验支持和文档整理。

---

## 2. 本次更新的核心内容

### 2.1 Hybrid FGate 核心实现落地

本次提交在 `vllm_eplb/vllm/distributed/eplb/eplb_state.py` 中正式引入了 `fgate-hybrid-cache` 的核心运行状态与执行路径，主要包括：

- local dynamic shadow bank 的元数据维护
- active/standby 双 bank 机制
- layer 级 local shadow logical id 选择
- async fast refresh 的 staging / consume 流程
- active shadow patched runtime map
- periodic static refresh 与 local dynamic refresh 的共存框架

这使得 hybrid 不再只是概念或配置层改动，而是具备了完整的状态机和运行路径。

### 2.2 Router 与模型前向路径支持 local shadow

在 `base_router.py`、`fused_moe/layer.py`、`qwen3_moe.py`、`deepseek_v2.py` 中，加入了 hybrid 所需的前向执行支持：

- 先做逻辑 expert 到 physical expert 的正常 EPLB 映射
- 再做本地 local dynamic shadow mapping
- 最后再记录 physical expert load

同时补上了：

- `consume_pending_layer_refresh()`
- `schedule_next_layer_shadow_refresh()`
- next-layer gate prediction 的 target layer 绑定

这让 hybrid 的“局部即时反应”能够真正插入 MoE 热路径。

### 2.3 Idea2 文档重写

`idea2.md` 被整体重写，不再沿用旧版本的混乱描述，而是明确区分：

- `EMA/static redundancy` 负责全局慢变量
- `FGate/local dynamic shadow` 负责本地快变量

新文档把 hybrid 的目标、对象、两阶段路由、即时更新语义和设计边界都重新定义清楚了。

---

## 3. 本次提交实际解决的主要问题

### 3.1 解决了 multi-DP 下 hybrid 周期性全局重排的卡死/失步问题

这是本次提交最关键的稳定性修复之一。

核心修复包括：

- 在 `gpu_worker.py` 中让 `execute_dummy_batch()` 不再只做 DP 对齐，而是继续执行 EPLB step
- 在 `gpu_model_runner.py` 中让 dummy batch 也走 `eplb_step(is_dummy=True)`
- 在 `eplb_state.py` 中让 periodic rearrange counter 在 real step 和 dummy step 上都前进
- 在 periodic rearrange 退出时显式加 barrier，保证各 rank 一致离开重排阶段

这解决了之前出现的：

- `out-of-order step`
- `No available shared memory broadcast block found in 60 seconds`
- `NCCL collective timeout`
- global rearrange 在不同 rank 提前/错位触发

### 3.2 解决了 hybrid 周期性重排与即时 refresh 共存时的稳定性冲突

本次提交新增了：

- `hybrid_periodic_rearrange_with_multi_dp`
- `_hybrid_enable_immediate_layer_refresh()`

当 `DP > 1` 且启用周期性全局重排时，会自动关闭 layer-level immediate refresh，并停止把 `expert_load_fgate` 下发到模型层，避免：

- 一边做 periodic global rearrange
- 一边在 decode 热路径做 layer-level shadow refresh

这个改动优先保证了：

- 吞吐稳定
- 全局重排可用
- hybrid 在多卡 serving 下不再因为两套机制互相打架而失稳

### 3.3 解决了 compile / cudagraph 与 hybrid 路径的兼容问题

在 `gpu_model_runner.py` 中，本次提交不再粗暴关闭 compile/cudagraph，而是改成更细粒度策略：

- `fgate-hybrid-cache` 强制切到 `PIECEWISE` cudagraph
- `fgate-peer-cache` 切到 `FULL_DECODE_ONLY`

同时在 forward 前增加 `prepare_for_forward()`，并在 hybrid refresh 相关路径中检查：

- `torch.compiler.is_compiling()`
- `torch.cuda.is_current_stream_capturing()`

这解决了此前出现的：

- CUDA graph capture 失败
- 在 capture 期间执行 refresh hook 导致报错
- 为了稳定性只能整体关闭 cudagraph、明显损失效率的问题

### 3.4 解决了 hybrid decode 额外开销过高的问题

本次提交新增了 `fgate_decode_stride`，并在 `forward_context.py` 中实现了 decode-step 采样与缩放逻辑：

- 不是每个 decode step 都必须跑完整 fgate predictor
- 被跳过的 step 用缩放补偿 load 统计

这降低了 hybrid decode 路径的额外前向开销，缓解了 TPOT/ITL 变差的问题。

### 3.5 解决了模型端 hybrid hook 与 load 统计接线不完整的问题

在 `qwen3_moe.py` 和 `deepseek_v2.py` 中，本次提交把 hybrid 需要的几条链路补齐了：

- consume pending refresh
- next-layer gate prediction
- per-layer fgate load accumulation
- schedule next layer refresh

此前这些链路不完整时，即使 EPLB state 里有 hybrid 逻辑，模型前向也无法真正触发。

### 3.6 解决了实验复现与公平 bench 的基础设施问题

脚本层面，本次提交新建或重写了：

- `scripts/serve_ema.sh`
- `scripts/serve_hybrid.sh`
- `scripts/bench_serve.sh`
- `scripts/chat.sh`

具体改善包括：

- EMA 与 hybrid server 分离启动
- server 常驻，不再每次 bench 都重新拉起
- bench 可以分别压测两个端口
- 增加 `SERVER_SEED`、`DATASET_SEED`、`REQUEST_SEED`
- bench 默认支持更公平的确定性配置

这解决了之前：

- benchmark 流程不稳定
- server 生命周期和 bench 绑死
- 输出不可复现、对比不公平
- 聊天脚本 JSON 非法、端口混乱的问题

### 3.7 解决了 domain-shift 数据准备困难的问题

本次提交加入了 `scripts/download_domain_shift_hf_datasets.py`，并将处理后的数据直接落到了 `data/domain_shift_hf/`：

- 8 个 domain 原始样本文件
- `stable` / `block_64` / `alternating_2` / `random_uniform` 四种 scenario
- `manifest.json`

脚本还额外处理了：

- hub 上脚本式数据集与文件式数据集的差异
- 某些 domain 样本数不足时的重采样
- 不同数据集 prompt 字段格式不统一的问题

这让后续高 domain-shift 压测有了标准输入。

---

## 4. 代码层面的主要改动点

### 4.1 EPLB / Hybrid 核心

- `vllm_eplb/vllm/distributed/eplb/eplb_state.py`
- `vllm_eplb/vllm/distributed/eplb/rebalance_execute.py`

主要负责 hybrid 状态、fast refresh、static refresh、周期性重排同步和本地 shadow map。

### 4.2 Router / Model 执行路径

- `vllm_eplb/vllm/model_executor/layers/fused_moe/router/base_router.py`
- `vllm_eplb/vllm/model_executor/layers/fused_moe/layer.py`
- `vllm_eplb/vllm/model_executor/models/qwen3_moe.py`
- `vllm_eplb/vllm/model_executor/models/deepseek_v2.py`
- `vllm_eplb/vllm/model_executor/models/qwen3_next.py`

主要负责：

- local dynamic shadow 二次分流
- next-layer fgate prediction
- per-layer refresh hook

### 4.3 Worker / Compile / Runtime

- `vllm_eplb/vllm/v1/worker/gpu_model_runner.py`
- `vllm_eplb/vllm/v1/worker/gpu_worker.py`
- `vllm_eplb/vllm/forward_context.py`
- `vllm_eplb/vllm/config/parallel.py`

主要负责：

- dummy step 对齐
- periodic rearrange 计数推进
- piecewise cudagraph 兼容
- decode stride 控制

### 4.4 脚本与实验支持

- `scripts/serve_ema.sh`
- `scripts/serve_hybrid.sh`
- `scripts/bench_serve.sh`
- `scripts/chat.sh`
- `scripts/download_domain_shift_hf_datasets.py`

### 4.5 文档与仓库管理

- `idea2.md`
- `.gitignore`

---

## 5. 本次提交带来的实验能力提升

在这次 commit 之后，项目已经具备：

- 单节点多卡下独立启动 EMA / hybrid server 的能力
- 用统一 domain-shift 数据集重复压测的能力
- 用固定 seed 做更公平对比的能力
- 在 multi-DP 环境下跑 hybrid 周期性全局重排而不立即卡死的能力

从工程角度看，这次提交把 hybrid 从“想法 + 零散 patch”推进到了“可运行、可复现实验、可继续调优”的阶段。

---

## 6. 当前版本仍需注意的点

虽然本次提交解决了大量稳定性问题，但从代码结构和实验阶段看，仍有几个现实边界：

- hybrid 的多 DP 稳定模式下，layer-level immediate refresh 不是始终开启的，需要根据是否启用 periodic global rearrange 来裁剪
- commit 中直接纳入了 `data/domain_shift_hf/` 大量实验数据，仓库体积会明显变大
- `results/` 目录虽然在当前工作区已加入 `.gitignore`，但历史上已被跟踪的结果文件不会自动取消跟踪

---

## 7. 总结

这次 `745238e` 提交完成的不是单点修补，而是一轮较完整的 hybrid fgate 工程化整合。它同时推进了三件事：

1. 把 `idea2` 从设计文档推进到代码实现
2. 把最严重的 multi-DP 周期性重排卡死问题修住
3. 把实验脚本、数据集和复现实验链路一并补齐

因此，这个 commit 的价值主要体现在：

- hybrid 终于具备了可运行的主干实现
- 原先阻塞实验的大部分系统性问题已经被绕开或修复
- 后续可以把重点从“能不能跑”转到“怎么继续提速和做更干净的对比”
