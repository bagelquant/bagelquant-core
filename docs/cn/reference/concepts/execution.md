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

## 边界

Execution 不负责读取市场数据、管理 provider 凭证、做组合回测或生成应用界面。它只负责 core 内部的图计算语义。
