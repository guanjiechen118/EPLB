# Idea2: Local Shadow FGate for Hybrid Cache

## 1. 目标

`idea2` 的核心目标，不是让 `fgate` 去频繁改全局 expert 布局，而是让它在 **decode 阶段** 做一件更直接、更及时的事情：

> 当某个 expert 在本 rank 上突然变成热点时，`fgate` 立即把一部分本地 token 分流到本地 shadow replica，缓解本地执行堵塞。

这里的关键词有三个：

- **本地**
- **即时**
- **只处理已经到达本 rank 的 token**

所以这版 `fgate` 本质上不是全局 EPLB，而是：

> `EMA` 负责全局慢变量，`FGate` 负责本地快变量。

---

## 2. 为什么原来的 hybrid 不对

原来的 `fgate-hybrid-cache` 动态部分，本质上还是在做：

- 改 `physical_to_logical_map`
- 改 `logical_to_physical_map`
- 改 `logical_replica_count`
- 让 dynamic bank 变成“全局可见副本”

这样会带来一个根本问题：

> 只要 dynamic refresh 改的是全局 runtime map，它就要求所有 rank 在同一个逻辑时刻切换到一致的新映射。

但在真实 decode 里，尤其是 `DP > 1` 时：

- 不同 rank 的请求进度不完全同步
- 不同 rank 的 decode step 也不完全同步

于是这个“每步动态切全局 map”的设计天然不稳。

这也是为什么原来的实现里，即时 refresh 在很多情况下必须退化、受限，或者只能做成比较保守的同步方案。

---

## 3. 正确的 idea2 应该是什么

正确的 `idea2` 应该把 hybrid 拆成两个时间尺度。

### 3.1 慢路径：EMA / static redundancy

这一部分负责：

- 全局长期热点
- 周期性静态刷新
- 真正意义上的全局 expert 复制与布局调整

这部分仍然可以：

- 改全局 `physical_to_logical_map`
- 改全局 `logical_to_physical_map`
- 做正常的 expert weight rearrangement

也就是说：

> `EMA/static` 负责“哪些副本应该长期存在于哪里”。

### 3.2 快路径：FGate / local dynamic shadow

这一部分负责：

- decode 阶段的短期热点
- 本地 rank 上的热点疏导
- 只影响本 rank 的 token 执行

它不应该：

- 改全局 runtime routing map
- 暴露给远端 rank
- 要求所有 rank 同步切换

也就是说：

> `FGate/dynamic` 负责“本地当前这批 token 要不要临时分流到 shadow replica”。

---

## 4. 本设计的核心对象

在 `fgate-hybrid-cache` 中，冗余资源被分成两类。

### 4.1 Static slots

由 `EMA` 管理。

特点：

- 全局可见
- 慢速刷新
- 负责长期热点
- 参与全局 `logical_to_physical_map`

### 4.2 Dynamic banks

由 `FGate` 管理。

特点：

- 只在本地 rank 生效
- 双 buffer
- 快速刷新
- 不参与全局 runtime map

它们更准确的名字应该是：

- `local dynamic shadow bank A`
- `local dynamic shadow bank B`

它们不是“新的全局 replica”，而是：

> 本地用于分担热点 expert 的 shadow replicas。

---

## 5. local shadow 的精确定义

对某一层 `l` 来说：

- 某些 physical expert 是 primary/home experts
- 某些 physical expert 是 static redundant experts
- 某些 physical expert 是 local dynamic shadow slots

其中 local dynamic shadow slots 的权重内容，在某个时刻会被设置成：

> 本 rank 上某个热点 logical expert 的一份本地 shadow 拷贝

但这些 shadow slot：

- 不会出现在全局的 `logical_to_physical_map`
- 不会被远端 rank 当作可 dispatch 目标
- 只在 token 已经落到本 rank 之后，才参与第二次本地分流

所以它们解决的是：

> post-dispatch 的本地执行拥塞

而不是：

> pre-dispatch 的全局通信负载不均衡

---

## 6. 两阶段路由

这版 `idea2` 的路由实际上分成两段。

### 第一阶段：正常全局路由

router 先按正常的：

- gate topk
- EPLB logical -> physical 映射
- static replicas

得到一个正常的 physical expert id。

这一步的输出仍然是“标准 EPLB 路由结果”。

### 第二阶段：本地 shadow 二次分流

然后只对下面这部分 token 做处理：

- 当前 expert 已经在本 rank 上
- 当前 logical expert 在 active local shadow bank 中有 shadow 副本

对于这些 token，可以把一部分从：

- `home/local physical expert`

改写到：

- `local dynamic shadow physical expert`

从而把一个本地热点 expert 的 token 压力拆开。

所以真正的执行语义是：

> 先全局决定 token 去哪个 rank，再在 rank 内部决定 token 去哪个 local replica。

---

## 7. 什么叫“即时更新”

这里的“即时”不是指：

> 当前 forward 进行到一半时，立刻修改这一步的执行目标。

当前设计的“即时”更准确地说是：

> decode 每一步都可以触发一次 local shadow refresh，并在下一次 forward 前生效。

也就是说：

1. 本步 decode 结束后，收集本步 `fgate` 预测
2. 立刻计算本地 dynamic shadow 的下一版分配
3. 异步把 standby bank 的权重准备好
4. 在下一次 forward 开始前完成 bank flip
5. 下一步 decode 立刻使用新的 active shadow bank

所以它是：

- **decode-step 级别**
- **next-step 生效**
- **本地即时**

而不是：

- 同一步中途热切换
- 跨 rank 同步热切换

这已经足够体现 `fgate` 相对 `EMA` 的核心优势：

> 它能以 decode step 为粒度，持续响应短期 domain shift。

---

## 8. 为什么这正适合高 domain shift 场景

当输入 domain 变化频繁时，热点 expert 可能表现为：

- 持续时间不长
- 切换很快
- 具有明显的局部突发性

这种情况下：

- `EMA` 太慢，容易跟不上
- 全局 remap 太重，成本太高
- 但本地执行瓶颈又确实会被放大

而 local shadow fgate 正好适合这个场景，因为它：

- 不需要改全局布局
- 不需要全局同步
- 每步都能响应
- 只对真正造成局部堵塞的热点进行疏导

所以这版 `idea2` 的优势，不在于“重新做一次全局负载均衡”，而在于：

> 用极小的局部动作，快速吸收高频 domain shift 带来的本地瞬时热点。

---

## 9. 双 buffer 为什么必要

dynamic shadow 之所以要用双 bank，而不是单 bank，是因为要避免：

- 当前计算使用中的 bank
- 下一轮要更新权重的 bank

发生冲突。

因此需要：

- `active bank`
- `standby bank`

典型流程是：

1. 当前 step 使用 `active bank`
2. 同时异步把下一版热点 expert 权重搬到 `standby bank`
3. 下一次 forward 前切换 active/standby

这样做的意义是：

- 当前计算不被正在写入的 bank 干扰
- refresh 和执行可以重叠
- 切换动作只发生在 safe point

---

## 10. 这版设计故意不解决什么

这很重要。

这版 `idea2` **故意不解决** 以下问题：

### 10.1 不解决全局 all-to-all 不均衡

因为 token 先按正常全局路由 dispatch。

如果某个 logical expert 的 token 从一开始就大量涌向某个 rank，那么：

- 通信负载已经发生了
- local shadow 只能缓解 rank 内执行瓶颈
- 不能消除已经发生的跨 rank 通信代价

### 10.2 不解决多 rank 同步 refresh

因为它本来就不该依赖这个。

### 10.3 不试图把 dynamic bank 暴露成新的全局 replica

因为一旦这么做，就又会回到旧问题：

- 全局 map 切换
- rank 间同步
- decode step 不一致

---

## 11. 当前实现原则

按现在的理解，正确实现应遵守下面几条原则。

### 原则 1

`static` 可以改全局 map，`dynamic` 不可以。

### 原则 2

dynamic bank 的权重更新可以异步，但生效边界必须在 forward safe point。

### 原则 3

router 的本地二次分流只作用于：

- 已经路由到本地的 token
- 当前 active shadow bank 已覆盖的 logical expert

### 原则 4

每一步最多刷新少量 layer，防止 refresh 开销反过来压垮 decode。

### 原则 5

prefill 不追求这条快路径，重点只在 decode。

---

## 12. 与纯 EMA、peer-cache 的区别

### 纯 EMA

- 只处理慢变量
- 适合长期热点
- 不擅长高频切换

### fgate-peer-cache

- 目标仍然更偏全局 replica 调整
- 更强调跨 rank / 跨 slot 的动态副本利用

### 本版 fgate-hybrid-cache local shadow

- static 用 EMA
- dynamic 用 local shadow
- dynamic 不改全局 runtime map
- dynamic 只解决本地热点 expert 堵塞

所以它更准确地说是：

> `EMA global residency + FGate local execution relief`

---

## 13. 最终定义

一句话总结这版 `idea2`：

> `fgate-hybrid-cache` 的 dynamic 部分不应该再被理解为“全局动态副本切换”，而应该被理解为“decode 阶段、本地 rank 内、面向热点 expert 堵塞的 local shadow immediate refresh”。

再压缩一点：

> `EMA` 决定长期放什么，`FGate` 决定下一步在本地帮谁分流。

这就是这版 `idea2` 最核心的设计思想。

---

## 14. 实现修正：真正符合 idea2 的执行时序

上面第 7 节把“即时更新”描述成：

- 本步 decode 结束后统计
- 下一次 forward 前完成 shadow refresh
- 下一步 decode 生效

这其实还不够“即时”。

如果严格按 `idea2` 的原始目标，正确实现应该更进一步：

> `fgate` 必须在 **每一个 layer** 之后立刻做出反应，用当前 layer 的预测去服务 **下一个 layer**，而不是等到整个 step 结束后再统一处理。

也就是说，真正应该采用的是 **layer pipeline**，不是 **step pipeline**。

### 14.1 正确的时序

对于第 `l` 层来说：

1. 进入第 `l` 层前：
   - 先消费上一个 layer 为当前层准备好的 `pending shadow refresh`
   - 如果当前层的 standby bank 已经准备好，就在这里切换成新的 active bank

2. 执行第 `l` 层时：
   - 正常完成本层 router + MoE 计算
   - 同时用本层 hidden state 经过 `next_gate_weight`，预测第 `l+1` 层的 expert load

3. 第 `l` 层预测一出来：
   - 立刻根据预测结果，为第 `l+1` 层选择新的 local shadow logical experts
   - 立刻发起第 `l+1` 层 standby bank 的本地 expert copy

4. 等执行流走到第 `l+1` 层入口时：
   - 正好尝试消费刚才异步准备好的 refresh
   - 如果 copy 已完成，则第 `l+1` 层直接使用新的 active shadow bank

所以它的目标不是：

> 当前 step 结束后，下一步再调

而是：

> 当前层刚预测完，立刻为下一层做准备

### 14.2 想要掩盖的到底是什么时间

这版设计最关键的工程目标是：

> 用当前层 MoE/MLP 的执行时间，掩盖下一层 local shadow copy 的准备时间。

也就是把：

- `layer l` 的预测
- `layer l+1` 的 shadow copy
- `layer l` 到 `layer l+1` 之间的正常计算

尽量做成流水化重叠。

这也是为什么 dynamic bank 必须是双 buffer：

- 当前层/当前轮正在使用 `active bank`
- 后台线程同时往 `standby bank` 填下一层要用的热点副本
- 到下一层入口再安全切换

### 14.3 这意味着什么

这意味着 `idea2` 的“即时”应该被重新定义为：

> **layer 级即时**

而不是：

> step 级即时

两者差别很大。

#### step 级即时

- 当前 step 全部 layers 跑完
- 才开始 refresh
- 下一步才生效

#### layer 级即时

- 当前 layer 刚预测完
- 立刻为下一层发起 refresh
- 下一层入口就尝试生效

后者才真正符合 `idea2` 的初衷。

### 14.4 为什么这比“step 尾部 refresh”更对

因为 `idea2` 的根本目标不是：

> 做一个比 EMA 更快一点的 step-level local refresh

而是：

> 让 `fgate` 成为 decode 过程中真正在线的、细粒度的本地热点响应机制

如果 refresh 只能发生在 step 尾部，那么它仍然太像“微型版 EMA”。

只有当 refresh 能插进 layer 间隙时，它才真正体现出：

- `fgate` 的预测性
- `fgate` 的本地性
- `fgate` 的即时性

### 14.5 对 prefill 的修正理解

第 11 节原来写的是：

> prefill 不追求这条快路径，重点只在 decode

这个表述需要修正。

更准确的说法应该是：

- `idea2` 的主要收益目标仍然在 **decode**
- 但从机制上说，`fgate` 的 **next-layer prediction** 不必强制跳过 prefill
- 也就是说，prefill 阶段可以允许做 prediction
- 只是最终是否有明显收益，要看 copy 能否被当前层计算时间掩盖

所以正确表述应为：

> 收益重点在 decode，但 `fgate` prediction 机制本身不必硬性排除 prefill。

### 14.6 对编译 / cudagraph 的含义

如果要支持这种 **layer-by-layer immediate reaction**，那么执行流必须允许在层与层之间回到 Python/runtime 控制逻辑。

因此这版设计天然不适合：

- 把整个 decode forward 完整包成一个不可打断的 full graph replay

否则：

- 当前 layer 结束后没法插入“为下一层发起 refresh”的控制逻辑
- 下一层入口也没法插入“消费 pending refresh”的 safe point

所以这版 `idea2` 在工程实现上要求：

> layer 间必须保留可执行的 runtime hook。

这是实现真正即时性的必要条件，不是可选优化。

### 14.7 最终修正后的定义

因此，最终版 `idea2` 应该这样表述：

> `EMA` 负责周期性的全局静态重排；`FGate` 负责每个 layer 对下一个 layer 的本地热点预测与 local shadow refresh。

再进一步压缩：

> 当前层预测下一层，当前层掩盖下一层 copy，下一层入口立即尝试生效。

这才是这版 `idea2` 最准确、最严格的执行语义。
