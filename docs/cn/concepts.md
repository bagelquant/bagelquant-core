# 核心概念

`bagelquant-core` 把量化研究逻辑拆成两部分：带研究域约束的数据对象，以及惰性执行的运算图。

## Domain

`Domain` 定义研究所使用的交易日历和资产空间。它不负责下载交易日历或证券主数据；调用方需要从数据层传入有序日历，以及静态资产列表或动态资产池。动态资产池使用包含 `time`、`asset_id` 和 `active` 的 long-form 布尔表；缺失成员关系视为非活跃。

## Panel

`Panel` 是与 `Domain` 对齐的不可变二维数值数据，形状是时间乘资产。原始输入、图的叶子节点、执行后的输出，以及执行过程中的缓存值都可以是 `Panel`。

参见 [Panel](reference/concepts/panel.md)。

`PredictionPanel` 是只能由 `PredictionComposer` 生成的强类型终端 subtype。普通 composer
始终输出普通 `Panel`，表示 AlphaValue，而不是 prediction。

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

## Prediction composer

Prediction 构建是 allowlist 中的终端 graph operation。Identity、equal-weight、rolling
positive-IC、rolling positive quantile-rank-IC、rolling OLS 与 rolling GLS 消费已经由调用方 Alpha Policy 完成日期对齐和
standardization 的 AlphaValue Panel。监督式 composer 接收通用 target 与 availability
Panel；core 不持有价格、schedule、standardization 或回测 policy。Window 长度由调用方
准备的 period 计数，缺失值不会 forward-fill。

`QuantileICWeightedPredictionComposer(window, quantiles)` 先用截面内所有有限 Alpha
值确定等人数分组，不会因 target 缺失而重划边界；随后计算高到低 quantile 分数与各组
target 均值的 Spearman 相关，并使用完整窗口平均 IC 的正值部分。Graph identity 同时
序列化 `window` 与 `quantiles`。

对于由应用层编排的学习流程，`ElasticNetPredictionComposer` 会完整序列化 walk-forward、
coverage、target、validation 与 Elastic Net contract，同时不依赖回测包。
`ZeroPreservingRmsScaler` 使用不中心化的加权 RMS，因此门控产生的零值在缩放后仍为零。
相对正则路径的候选只使用每折 sample 计算 `alpha_max`；sample+validation 重训时会重新
计算 `alpha_max`，但保留已经选中的 ratio。

## Execution

执行过程会校验图结构，按拓扑顺序计算依赖，缓存中间结果，并重新应用动态成分约束，避免非活跃资产单元进入后续计算。

参见 [Execution](reference/concepts/execution.md)。
