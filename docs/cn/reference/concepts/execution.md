# Execution

Execution 负责把惰性图计算成物化的 `Panel` 输出。

## 执行流程

执行时，运行时会：

- 校验图结构和依赖
- 按拓扑顺序计算父节点
- 对每个操作调用底层 Polars 函数
- 缓存中间输出
- 重新应用 `Domain` 的动态资产池屏蔽

## 输出访问

图执行前不能读取 `Graph.output`。调用 `compute()` 后，输出节点以及参与计算的中间节点都会持有对应的 panel 输出。

## 边界

Execution 不负责读取市场数据、管理 provider 凭证、做组合回测或生成应用界面。它只负责 core 内部的图计算语义。
