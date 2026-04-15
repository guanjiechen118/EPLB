# Contribution / 汇报材料

## 0. 总结

这轮工作的主线非常明确：

1. 把 `idea2` 的设计思想从“概念”收敛成可以落地的工程语义。
2. 把 `fgate-only` 从“还能跑，但路径里有明显多余成本”的状态，推进到“更像一个真正的 decode-oriented local shadow 机制”。
3. 把 prefill 和 decode 的职责显式分离，避免 `fgate-only` 在 prefill 阶段反向伤害整体性能。
4. 补齐 benchmark、配置开关、单测，形成可复现实验闭环。

一句话概括当前分支的方向：

> `EMA` 负责慢速、全局、长期热点；`FGate` 负责快速、本地、短期热点；`prefill` 尽量贴近 baseline，`decode` 才真正吃到 local shadow 的收益。

---

## 1. 今日落地的优化

### 1.1 FGate predictor 计算路径瘦身

#### 背景

之前 `fgate` 在每个 MoE layer 里会额外做一次：

- `next_gate linear`
- `softmax`
- `sum(dim=0)`

这条路径完全在主前向路径上，不能简单认为“可忽略”。在 `fgate-only` 场景里，它本身就是相对 baseline 的主要残余成本来源之一。

#### 今日改动

新增了统一工具文件：

- `vllm_eplb/vllm/model_executor/models/moe_fgate_utils.py`

其中实现了三件事：

1. `project_with_linear_weight(...)`
   负责 predictor 的统一线性投影。

2. `maybe_fused_gate_and_next_logits(...)`
   当 gate 是普通未量化线性层时，把：
   - 当前层真实 gate
   - 下一层 predictor gate

   拼成一次 GEMM。

3. `predicted_load_from_topk(...)`
   直接基于 `pred_logits` 做 `top-k` 热点统计，不再走全量 `softmax + sum`。

#### 涉及文件

- `vllm_eplb/vllm/model_executor/models/moe_fgate_utils.py`
- `vllm_eplb/vllm/model_executor/models/qwen3_moe.py`
- `vllm_eplb/vllm/model_executor/models/deepseek_v2.py`
- `vllm_eplb/vllm/model_executor/models/qwen3_next.py`

#### 设计意义

这一步本质上做了两层优化：

1. **减少额外 GEMM 次数**

原先是：

- 当前 gate 一次
- predictor 再一次

现在在可融合路径上变成：

- 一次拼接 GEMM

2. **把预测目标从“完整概率分布”改成“热点 expert 集合”**

EPLB 的目标不是恢复精确概率，而是识别热点 expert 并决定 shadow bank 该给谁。

因此从：

- `softmax -> 期望负载`

改成：

- `top-k -> 命中计数`

更符合当前算法目标。

#### 当前收益判断

这一步不会让 predictor 成本“消失”，但能明显降低：

- 额外线性层开销
- full softmax 开销

它属于 **真正打到当前残余开销根因** 的优化，而不是参数微调。

#### 当前边界

`qwen3_next` 的 internal-router 路径没有强行做外部 GEMM fusion，原因是那条路径和 `SharedFusedMoE` 的内部 router 计算耦合更紧，贸然改容易重复计算或破坏原语义。当前该分支只吃到了“top-k 直方图替代 softmax”的收益。

---

### 1.2 `fgate-only` 不再为死路径分配 `expert_load_fgate`

#### 背景

之前 `fgate-only` 在状态上还保留了一些更像 `hybrid` 遗留的张量路径，尤其是 `expert_load_fgate` 的分配语义并不干净。

#### 今日改动

在 `EplbState.add_model()` 中做了语义收敛：

- `fgate-only` 不再分配全局 `expert_load_fgate`
- 但会显式打开 `enable_next_gate_prediction`

涉及文件：

- `vllm_eplb/vllm/distributed/eplb/eplb_state.py`
- `vllm_eplb/vllm/model_executor/models/interfaces.py`
- 多个 `MixtureOfExperts.set_eplb_state(...)` 实现文件

#### 设计意义

这一步实际上把两个概念彻底分开了：

1. `hybrid` 需要的全局逻辑负载张量
2. `fgate-only` 需要的“下一层预测能力”

也就是说：

> `fgate-only` 要的是 predictor，不是 hybrid 那套全局 fgate 统计状态。

这让 `fgate-only` 的实现语义更纯，也为后续继续优化 predictor 路径打下了基础。

---

### 1.3 `fgate_skip_prefill` 配置化

#### 背景

实测上已经观察到：

> `fgate-only` 只在 decode 使用时效果较好；一旦 prefill 也开启 `fgate-only`，整体效果会明显变差。

这个现象和当前实现完全一致：

- prefill 不吃 decode stride
- predictor 成本在 prefill 上按 token 数直接放大
- refresh 更难被层间隙掩盖

#### 今日改动

新增配置项：

- `fgate_skip_prefill: bool = True`

涉及文件：

- `vllm_eplb/vllm/config/parallel.py`
- `vllm_eplb/vllm/distributed/eplb/eplb_state.py`
- 多个模型的 `set_eplb_state(...)`
- `start_server_fgate_only.sh`

#### 设计意义

这一步把“是否在 prefill 上做 fgate”从硬编码行为变成了显式实验变量：

- 默认仍然保持 decode-only
- 需要时可以做对照实验

这对汇报和后续 ablation 都很重要，因为它把一个关键结论从“口头经验”变成了“可控配置”。

---

### 1.4 `fgate_prefill_ignore_redundant`：让 prefill 尽量接近无 EPLB 路径

#### 背景

即使 `fgate_skip_prefill=true`，prefill 也不等于“完全无 EPLB”：

- router 仍会走 `logical_to_physical_map`
- 仍可能命中 redundant replica
- 仍可能做 local shadow rewrite
- 层入口仍可能消费 pending refresh

所以“skip predictor”并不能真正把 prefill 拉回 baseline。

#### 今日改动

新增配置项：

- `fgate_prefill_ignore_redundant: bool = False`

开启后，prefill 会：

1. 只走 primary expert mapping
2. 强制 `logical_replica_count = 1`
3. 跳过 local dynamic shadow rewrite
4. 跳过 prefill 路径的 expert-load bookkeeping
5. 跳过 prefill 层入口的 `consume_pending_layer_refresh()`

涉及文件：

- `vllm_eplb/vllm/config/parallel.py`
- `vllm_eplb/vllm/distributed/eplb/eplb_state.py`
- `vllm_eplb/vllm/forward_context.py`
- `vllm_eplb/vllm/model_executor/layers/fused_moe/layer.py`
- `vllm_eplb/vllm/model_executor/layers/fused_moe/router/base_router.py`
- `vllm_eplb/vllm/model_executor/models/qwen3_moe.py`
- `vllm_eplb/vllm/model_executor/models/deepseek_v2.py`
- `vllm_eplb/vllm/model_executor/models/qwen3_next.py`
- `start_server_fgate_only.sh`

#### 设计意义

这一步是非常关键的工程分离：

- `decode` 继续使用 runtime map + local shadow
- `prefill` 尽量回到 primary-only 的简单路径

也就是说：

> prefill 负责“尽量别拖慢首包”，decode 才负责“局部热点执行疏导”。

#### 对 decode 的影响

这个开关只在：

- `is_prefill_batch`
- 且 `fgate_skip_prefill=true`

时生效。

decode 主路径没有改 routing 结构，因此不会把 decode 优化收益反向损伤掉。

---

### 1.5 refresh copy 路径从逐行 copy 改成 batched row copy

#### 背景

在 local shadow refresh 的实现里，原先有多处是：

- 先逐个目标 row 找 source row
- 再逐行 `.copy_()`

这种写法的问题有两个：

1. Python 层循环太重
2. 对 duplicate rows 的处理不够优雅

#### 今日改动

在 `rebalance_execute.py` 中新增：

- `_resolve_source_rows(...)`
- `_copy_rows_batched(...)`

把以下路径统一成批量索引 copy：

- `move_to_buffer(...)`
- `move_to_buffer_local_only(...)`
- `move_from_buffer(...)`

涉及文件：

- `vllm_eplb/vllm/distributed/eplb/rebalance_execute.py`

#### 设计意义

这一步优化的目标不是“改变算法”，而是降低 local shadow refresh 的执行成本，使其更符合 `idea2` 对“本地即时刷新”的要求。

它同时带来两个价值：

1. **性能优化**
   减少 Python 循环和小颗粒度 copy。

2. **正确性增强**
   duplicate destination row 的处理更清晰，避免局部 shadow slot 重复目标时出错。

#### 补充说明：refresh 前的“预读取/预取数”也尽量放到了 GPU

这一点在之前的口头讨论里提过，但文档里原来没有单独写明，这里补充进去。

这轮对 `rebalance_execute.py` 的改动，不只是把“逐行 copy”改成了“批量 copy”，还把真正的数据读写路径尽量收敛到了 GPU 侧：

- 先在 CPU 上完成必要的轻量索引解析
  - 例如用 `_resolve_source_rows(...)` 找到 source row 和 target row 的对应关系
- 然后把 row index 转成 tensor，并搬到对应 device
- 再用 `index_select` / `index_copy_` 在 GPU 上完成批量 gather 和批量写回

也就是说，优化目标不是单纯“少写几个 Python for 循环”，而是：

> 在 refresh 前的预读取阶段，就尽量把真正的行读取和行拷贝放到 GPU 上执行，减少 CPU 逐行参与和细粒度调度开销。

从工程角度看，这一点很重要，因为 `idea2` 的 local shadow refresh 是高频短路径：

- 如果预读取阶段仍然严重依赖 CPU 逐行处理
- 那么 refresh 自己就可能成为 decode 路径的新瓶颈

所以这部分工作的实质可以概括成：

> 既做了 batched copy，也做了 GPU-side gather / pre-read，目标是尽量把 refresh 变成一条更短、更适合被 layer 间隙掩盖的路径。

---

### 1.6 benchmark 工具链收敛

#### 背景

如果 benchmark 配置本身不稳定，就很难准确比较 EPLB 算法。

在这轮讨论中，已经形成几个清晰结论：

- `request-rate=inf` 更适合测最大吞吐和算法纯性能
- `burstiness` 在 `request-rate=inf` 下没有作用
- `temperature=0` 更适合做严格对比，因为它减少采样随机性和路由轨迹波动
- 对 `fgate-only`，更容易拉开差距的是 decode-heavy workload

#### 今日改动

1. 新增 `run_benchmark_random.sh`
   - 基于 `random` dataset
   - 默认更偏 decode-heavy
   - 默认扫多档 `max_concurrency`

2. 调整 `run_benchmark.sh`
   - 使用 `request-rate=inf`
   - 固定 `seed=0`
   - 固定 `temperature=0`

3. 扩展 `start_server_fgate_only.sh`
   - 暴露 `FGATE_SKIP_PREFILL`
   - 暴露 `FGATE_PREFILL_IGNORE_REDUNDANT`

#### 设计意义

这部分不是算法优化本身，但它是算法评估质量的前提。没有稳定的 benchmark 方法，后面的结论都不够扎实。

---

### 1.7 单测和验证补齐

#### 新增/更新测试

1. `vllm_eplb/tests/model_executor/test_moe_fgate_utils.py`
   - 验证 fused gate + next_gate 与两次独立 linear 一致
   - 验证非未量化 gate 自动 fallback
   - 验证 top-k 负载统计逻辑

2. `vllm_eplb/tests/model_executor/test_eplb_prefill_router.py`
   - 验证 prefill primary-only mapping
   - 验证 decode 路径不受影响

3. `vllm_eplb/tests/test_eplb_peer_cache.py`
   - 验证 `fgate_skip_prefill`
   - 验证 `fgate_prefill_ignore_redundant`
   - 验证 `fgate-only` 不再依赖 dead `expert_load_fgate`
   - 验证 batched local copy / duplicate row 行为

#### 验证方式

本轮至少补齐了三类验证：

- `compileall`
- `bash -n` 脚本检查
- `pytest` 单测

这意味着当前优化不是只停留在“代码改了”，而是已经具备可回归性。

---

## 2. 基于 `idea2.md` 的整体设计修正

这一部分不是“某个单独 commit”，而是本轮工作对 `fgate` 路线的总设计认识。做汇报时，建议把它作为“架构层面的核心结论”讲清楚。

### 2.1 `idea2` 的核心不是全局动态重排，而是本地 local shadow

`idea2.md` 反复强调的核心点是：

> `FGate` 不应该继续承担“频繁修改全局 runtime map”的角色，而应该变成一个 decode 阶段、本地 rank 内、针对热点 expert 堵塞的 local shadow refresh 机制。

也就是说：

- 原来的错误方向是：把 dynamic refresh 当成“更快的全局 map 切换”
- 正确方向是：把 dynamic refresh 当成“本地执行热点疏导”

这是整个路线最重要的语义修正。

---

### 2.2 慢路径和快路径必须分开

`idea2` 给出的标准拆法是：

1. **慢路径：EMA / static redundancy**
   - 全局可见
   - 周期性静态刷新
   - 处理长期热点

2. **快路径：FGate / local dynamic shadow**
   - 只在本 rank 生效
   - 双 buffer
   - 高频刷新
   - 处理短期热点

换句话说：

> `EMA` 负责“长期放什么”，`FGate` 负责“下一步帮谁分流”。

---

### 2.3 local shadow 解决的是 post-dispatch 执行拥塞

`idea2` 明确指出，dynamic bank 不应该被暴露成新的全局 replica。

它们的语义是：

- token 已经按正常全局路由到本 rank
- 在本 rank 内部，再把一部分 token 从 home expert 分流到 local shadow

所以它解决的是：

- rank 内热点 expert 的执行堵塞

而不是：

- 已经发生的 all-to-all 通信不均衡

这也是为什么 `fgate-only` 天然更擅长减轻局部执行瓶颈，而不是替代全局 EPLB。

---

### 2.4 真正正确的执行时序应该是 layer pipeline，不是 step pipeline

`idea2.md` 后半部分最重要的修正是：

> 真正的“即时更新”应该是 layer 级别，而不是 step 级别。

即：

1. 第 `l` 层执行时，预测第 `l+1` 层热点
2. 立刻为第 `l+1` 层发起 shadow refresh
3. 在第 `l+1` 层入口尝试消费 refresh

压缩成一句话就是：

> 当前层预测下一层，当前层掩盖下一层 copy，下一层入口立即尝试生效。

这也是为什么：

- dynamic bank 必须双 buffer
- layer 间必须保留 runtime hook
- full-graph replay 不适合这条路径

---

### 2.5 prefill 不是收益重点，但也不能简单理解成“完全无关”

`idea2` 对 prefill 的修正理解是：

- 主要收益目标仍然在 decode
- 但从机制上看，next-layer prediction 并不天然只能用于 decode

不过结合当前实测和当前实现，工程上的结论已经很清楚：

1. **默认值** 仍应以 decode-only 为主
2. prefill 的首要目标不是抢收益，而是尽量不拖慢 TTFT
3. 如果要在 prefill 上做探索，也应该更保守、更便宜，而不是直接照搬 decode 全量 predictor 路径

---

### 2.6 `idea2` 明确“不解决什么”

做汇报时非常建议把这个部分讲出来，因为它能避免对方案产生不合理预期。

当前路线**不试图**解决：

1. 全局 all-to-all 通信负载不均衡
2. 多 rank 同步的动态 refresh
3. 把 dynamic bank 变成新的全局 replica

也就是说，这条路线的目标不是“全局动态均衡一切”，而是：

> 用最小的本地动作，吸收高 domain shift 下的局部瞬时热点。

---

## 3. 此前已经完成的优化点汇总

这一节按“功能主题”汇总，适合直接拿来做汇报中的“工作包”。

### 3.1 FGate 的职责重新定义

已经完成的语义修正：

- `fgate-only` 不再被当成 mini-hybrid
- `fgate-only` 的核心职责变成：
  - next-layer prediction
  - local shadow refresh
  - decode 阶段局部热点缓解

这是整个路线从“概念混杂”走向“语义清晰”的第一步。

---

### 3.2 predictor 重新接回主路径

在当前代码状态下，`fgate-only` 的 predictor 路径已经明确接入模型侧：

- 模型层可拿到 `next_gate_weight`
- layer 内会根据 predictor 结果调度下一层 shadow refresh

这一步的意义是：

> `fgate-only` 终于不是只有配置名，而是有一条真正可执行的 next-layer predictor 路径。

---

### 3.3 decode stride 可控

`fgate-only` 当前已经支持 `fgate_decode_stride`，这是非常重要的一项工程杠杆。

作用：

- 控制 predictor 的执行频率
- 在“响应速度”和“额外开销”之间做权衡

意义：

- 让 `fgate-only` 从固定成本，变成可调成本
- 为不同 workload 提供对照实验基础

---

### 3.4 `fgate_skip_prefill` 默认 decode-only

这项优化的重要性不只是“加一个开关”，而是把一个已经被实验证明的现象固化成默认行为：

- `fgate-only` 默认只管 decode
- prefill 默认不走 predictor

这是当前最稳妥、也最符合性能观察的默认策略。

---

### 3.5 prefill primary-only 回退路径

即今天新增的 `fgate_prefill_ignore_redundant`。

从汇报口径上看，它并不是一项孤立 patch，而是此前 prefill 路线思考的自然延伸：

- 先确认 prefill 不应重度依赖 fgate
- 再确认只 skip predictor 还不够
- 最后落成 primary-only mapping + skip refresh consume

它标志着：

> prefill 和 decode 已经不再被强行绑在同一条 EPLB 执行语义上。

---

### 3.6 refresh copy 路径批量化

local shadow 的价值能否兑现，不只取决于 predictor 准不准，也取决于 refresh 路径本身够不够轻。

此前已经完成的关键工程优化就是：

- `move_to_buffer_local_only`
- `move_from_buffer`

从逐行 copy 改成 batched row copy。

这类优化虽然不显眼，但对 `idea2` 这种“高频、本地、短路径”机制至关重要，因为它直接决定 refresh 能不能被 layer 间隙掩盖。

---

### 3.7 benchmark 方法学收敛

此前已经形成的一套较稳定 benchmark 认知包括：

1. `request-rate=inf` 适合测最大吞吐
2. `burstiness` 在 `request-rate=inf` 下无作用
3. `temperature=0` 更适合做算法对比
4. 要放大 `fgate-only` 差异，应优先构造 decode-heavy workload
5. `max-concurrency` 比单纯提高 `request-rate` 更容易拉开 EPLB 算法差异

这套认知现在已经部分固化到脚本里，不再只是口头经验。

---

## 4. 关键实验观察与当前结论

这一节适合放在汇报中的“结论页”。

### 4.1 `fgate-only` 在 decode-only 时效果更合理

当前最清晰的实验观察是：

- `fgate-only` 只在 decode 上使用时，效果相对更好
- 一旦 prefill 也强行开启 `fgate-only`，整体性能会明显变差

这个现象和当前实现机制完全一致，原因包括：

1. prefill 不享受 decode stride 降频
2. predictor 成本按 prefill token 数被放大
3. refresh 更难被掩盖
4. prefill 上热点更散、更抖，shadow bank 收益不稳定

因此当前结论是：

> `fgate-only` 的默认适用域应当明确收敛在 decode。

---

### 4.2 predictor 成本不能忽略

当前已经形成的明确结论是：

- predictor 不是“白送”的
- `next_gate linear + scoring + aggregation` 在主路径上同步执行
- 它不能被简单理解为“后台成本”

因此，所有后续优化都应围绕：

1. 降 predictor 本身的计算量
2. 降 refresh 本身的 copy 成本
3. 让两者尽可能被 layer pipeline 掩盖

---

### 4.3 local shadow 的收益边界也很清楚

当前路线更擅长缓解：

- 本地 rank 上热点 expert 的执行堵塞

而不擅长解决：

- 全局 token dispatch 不均衡
- all-to-all 代价

这不是缺点，而是方案刻意选择的适用范围。

---

## 5. 当前仍然存在的限制

为了让汇报更完整，建议把这些限制直接讲明。

### 5.1 predictor 成本仍未被完全吃掉

即使做了 GEMM fusion 和 top-k 直方图化，predictor 仍然不是零成本。它只是从“明显重”变成了“更接近可接受”。

### 5.2 `qwen3_next` internal-router 路径还没完全融合

当前为了保持正确性，没有强行对 internal-router 分支做外部 GEMM fusion。这意味着该分支仍有继续优化空间。

### 5.3 `fgate-only` 仍然不解决全局通信负载问题

它只解决 local execution bottleneck，不解决 pre-dispatch 的全局 token 倾斜。

### 5.4 layer-level immediate reaction 对图编译不友好

`idea2` 的严格语义要求 layer 间可插入 runtime hook，这与“整段 decode 完整 graph replay”天然有张力。

---

## 6. 建议在汇报中怎么讲

建议把整个工作分成四页主线：

### 第一页：问题定义

- baseline vs EPLB 的真实 gap 不在“有没有功能”，而在“多出来的路径成本”
- `fgate-only` 的目标不是全局动态重排，而是本地热点执行疏导

### 第二页：设计修正

- `EMA` 管慢变量
- `FGate` 管快变量
- `static` 管长期驻留
- `dynamic shadow` 管本地即时分流

### 第三页：工程落地

- predictor 路径瘦身：GEMM fusion + top-k 统计
- prefill 路径收缩：skip predictor + ignore redundant
- refresh copy 批量化
- benchmark 与测试闭环

### 第四页：当前结论与后续

- decode-only 是当前最合理默认策略
- predictor 成本仍是后续重点
- local shadow 路线成立，但收益边界就是“本地执行热点”

---

## 7. 可直接引用的一句话版本

### 版本 A：偏技术

> 本轮优化把 `fgate-only` 从一条带明显额外开销的预测路径，收敛成了更符合 `idea2` 的 decode-oriented local shadow 机制：prefill 尽量贴近 baseline，decode 使用 next-layer prediction 驱动本地 shadow refresh，并通过 predictor 计算瘦身和 refresh copy 批量化降低残余成本。

### 版本 B：偏汇报

> 我们没有继续把 `FGate` 当成“更快的全局重排器”，而是把它重新定义成“本地热点执行疏导器”：`EMA` 负责全局长期热点，`FGate` 负责 decode 阶段下一层的本地即时分流。

### 版本 C：偏结果导向

> 当前分支最核心的贡献，是把 `fgate-only` 的收益集中到 decode，把 prefill 的负担尽量剥离出去，并围绕 predictor 和 refresh 两条真实成本路径做了针对性瘦身。
