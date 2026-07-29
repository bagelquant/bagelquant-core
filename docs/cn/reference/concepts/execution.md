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

## 边界

Execution 不负责读取市场数据、管理 provider 凭证、做组合回测或生成应用界面。它只负责 core 内部的图计算语义。
