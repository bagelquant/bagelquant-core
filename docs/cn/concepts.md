# 核心概念

`bagelquant-core` 把量化研究逻辑拆成两部分：带研究域约束的数据对象，以及惰性执行的运算图。

## Domain

`Domain` 定义研究所使用的交易日历和资产空间。它不负责下载交易日历或证券主数据；调用方需要从数据层传入有序日历，以及静态资产列表或动态成分布尔表。

## Panel

`Panel` 是与 `Domain` 对齐的不可变二维数值数据，形状是时间乘资产。原始输入、图的叶子节点、执行后的输出，以及执行过程中的缓存值都可以是 `Panel`。

参见 [Panel](reference/concepts/panel.md)。

## Graph

`Graph` 表示惰性的研究逻辑链。Transformer 和 composer 不会立刻计算结果，而是返回新的图节点；只有调用 `compute()` 时才会物化输出面板。

参见 [Graph](reference/concepts/graph.md)。

## Transformer

Transformer 是一元运算：

```text
Panel | Graph -> Graph
```

常见用途包括排序、标准化、滚动窗口、缺失值处理、极值处理，以及因子变换。

参见 [Transformer](reference/concepts/transformer.md)。

## Composer

Composer 用多个 `Panel` 或 `Graph` 输入生成一个新图，适合算术组合、分组截面运算、滚动关系、投影和加权聚合。

参见 [Composer](reference/concepts/composer.md)。

## Execution

执行过程会校验图结构，按拓扑顺序计算依赖，缓存中间结果，并重新应用动态成分约束，避免非活跃资产单元进入后续计算。

参见 [Execution](reference/concepts/execution.md)。
