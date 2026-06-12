# BagelQuant Core

`bagelquant-core` 是 BagelQuant 的共享研究内核，负责面板数据模型、惰性图逻辑、算子组合和执行运行时。

它适合在数据已经进入 pandas 或 `Panel` 之后使用：先把原始输入对齐到同一个研究域，再用 transformer 和 composer 定义因子、预测或权重逻辑，最后执行图得到可复现的结果。

## 推荐阅读

- [快速开始](quick-start.md)
- [核心概念](concepts.md)
- [架构与设计](architecture.md)
- [公开 API](reference/public-api.md)
- [内部实现](reference/internals.md)
- [Panel](reference/concepts/panel.md)
- [Graph](reference/concepts/graph.md)
- [Transformer](reference/concepts/transformer.md)
- [Composer](reference/concepts/composer.md)
- [Execution](reference/concepts/execution.md)
- [API Reference](reference/index.md)

