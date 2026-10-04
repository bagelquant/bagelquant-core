# Execution

Execution 负责把惰性图编译成稀疏 Polars 计划，并在必要边界物化为 `Panel` 输出。

## 执行流程

执行时，运行时会：

- 校验图结构和依赖
- 按拓扑顺序计算父节点
- 按 `OperationContract` 决定 lazy、稠密或 eager barrier
- 在计划中传播 lineage trace columns
- 缓存中间计划，并让共享子图只执行一次
- 多输出共同收集，最终按 `Domain` 对齐

## 输出访问

图执行前不能读取 `Graph.output`。重复批次应先调用 `Graph.compile(spec)`，
然后复用 `CompiledGraph` 和 `ExecutionRuntime`。Core 不提供磁盘缓存。

## 持久物化协议

应用实现 `MaterializationStore.query(key)` 和 `publish(value)`，再通过
`ExecutionRuntime(materialization_store=adapter)` 使用现有执行器。
`MaterializationLookup` 返回 HIT、PARTIAL 或 MISS；仅完整、已证明的 HIT 可直接
复用。PARTIAL 不会静默混入旧值；增量继续前须由应用证明完整因果输入前缀。

逻辑 ID 与执行 identity 独立。`MaterializationKey` 包含算子及运行时版本、相关
不可变输入收据、可用性 trace、Domain、已绑定参数和 checkpoint 上下文。
无关 Snapshot 更新不必使节点缓存失效。重建输入须提供稳定 `source_key`，
其 `identity` 和 `trace_identity` 应由不可变内容证明，不能使用可变文件名。

`NodeMaterialization` 保留 Panel 类型、稀疏坐标、trace、checkpoint、训练审计和
具名数值附件。`rebalance` 的完整目标和 hold/unavailable 决策一同保存与复用。
`runtime.node_materializations` 按逻辑节点 ID 暴露收据，`node_artifacts` 暴露附件。

运行时按资源预算分批收集待保存的稀疏节点，随后用存储适配器返回的不可变扫描计划
替换旧计划并释放 eager 数组。应用仍须提交受限日期块；资源限制不能缩短训练历史
或改变调仓节奏。数据库、数据版本、历史失效、原子发布及保留策略由应用负责。

## 有界并发和日期块

`ResourceLimits.parallel_nodes > 1` 时，运行时按就绪依赖前沿并发执行独立 eager
数值内核。lazy 计划仍由 Polars 共享线程池融合执行。共享 eager 输入共同收集；
每个线程有隔离的 checkpoint、训练审计和输出证据，主线程按稳定图顺序合并并发布。
`for_workers(实际准入数)` 分配总线程、LightGBM 线程和直方图、内存及临时数组预算。
应用在导入数值库前设置 Polars 池和 BLAS/OpenMP 池，后者须按最大并发内核数分额。

运行时采样当前 RSS，在压力下减少后续节点准入、缩小 batch_rows 并释放可选缓存。
rank 比较批次消费该预算；regression moments 保持固定算术分块，避免改变累加起点后
在同一数值键下产生浮点差异。`runtime.resource_usage` 记录观测内存、压力事件
和实际并发。内存目标仍是软限制；单个必要内核可能超出估计，应用必须限制日期块，
不能截掉其必要训练历史或改变数值语义。

`runtime.plan_materialization_keys(graph)` 不执行算子或收集输入，只规划完整物化键；
`runtime.node_domains` 给出对应 Domain，因此可先查证已有结果，再准备缺失的辅助数据。
`causal_history_requirements(logical_spec)` 累加已证明有限算子的前序观察数，未知或
无界依赖返回 None。观察数按每个资产的有效坐标计数；动态 Universe 的缺失日期需要
成员记录证明 halo，不能直接把观察数当交易日数。完整输入前缀证明、stateful 恢复、
日期块裁剪与完整收据发布仍由应用负责。

新物化收据使用 `materialization_trace_identity`，其 trace token 由完整物化键和列名
决定，与日期块、halo 和作者名称无关，完整执行和有证明的块执行共用该 token。

`ExecutionRuntime(node_contexts={逻辑节点ID: context版本})` 可为选定节点指定数值
执行政策，其他节点使用默认 context_identity。规划与执行共用该映射，父节点键的
变化会传递到所有消费者。应用可组合固定有限块政策与完整历史下游内核；硬件预算
不能进入这些 context 版本。

## 边界

Execution 不负责读取市场数据、管理 provider 凭证、做组合回测或生成应用界面。它只负责 core 内部的图计算语义。
