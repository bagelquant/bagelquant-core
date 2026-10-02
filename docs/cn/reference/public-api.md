# 公开 API

稳定 API 主要从 `bagelquant_core`、`bagelquant_core.transformer` 和 `bagelquant_core.composer` 导出。

## 顶层对象

- `Domain`：交易日历和静态或动态资产池。
- `Panel`：按 `Domain` 对齐的不可变数值面板。
- `CategoryPanel`：按 `Domain` 对齐的不可变标签面板。
- `Graph`：由 transformer 和 composer 产生的惰性逻辑。

```python
from bagelquant_core import CategoryPanel, Domain, Graph, Panel
```

## Transformer

Transformer 接收一个 `Panel` 或 `Graph`，返回一个 `Graph`。

```python
from bagelquant_core.transformer import rank, winsorize, zscore

factor = rank(zscore(winsorize(raw_panel)), name="factor")
```

完整列表见 [Transformer reference](../../en/reference/transformers/index.md)。

## Composer

Composer 接收一个或多个 `Panel` 或 `Graph`，返回一个 `Graph`。

```python
from bagelquant_core.composer import div, weighted_sum

ratio = div(book, price, name="book_to_price")
prediction = weighted_sum(ratio, quality, weights=[0.6, 0.4])
```

完整列表见 [Composer reference](../../en/reference/composers/index.md)。

## Prediction Composer

强类型 Prediction 构建从 `bagelquant_core` 导出。生成的 `PredictionPanel` 可以经过
Transformer 链并保持类型，但不能进入另一个 Composer 或命名 Panel 参数。Quantile rank
IC 数值函数和 composer 共用同一公开口径：

```python
from bagelquant_core import (
    ICWeightedDecayPredictionComposer,
    QuantileICWeightedPredictionComposer,
    quantile_rank_information_coefficient,
)

composer = QuantileICWeightedPredictionComposer(window=12, quantiles=10)
decayed = ICWeightedDecayPredictionComposer(window=12, half_life=6)
```

`q1` 包含最高 Alpha 值。若 q1 到 qN 的 target 收益严格递减，quantile rank IC 为
`+1`；分组不完整或组收益完全相同则不产生 IC。

衰减 Composer 沿用完整窗口和正 IC 规则；经过调用方指定的 `half_life` 个 period 后，
旧 IC 观测的权重是最新观测的一半。

## 自定义操作

项目内逻辑可以用装饰器注册为与内置操作一致的函数。

```python
import polars as pl

from bagelquant_core.composer import composer
from bagelquant_core.transformer import transformer


@transformer
def demean(frame: pl.DataFrame) -> pl.DataFrame:
    means = frame.group_by("time").agg(pl.col("value").mean().alias("mean"))
    return (
        frame.join(means, on="time")
        .with_columns((pl.col("value") - pl.col("mean")).alias("value"))
        .select("time", "asset_id", "value")
    )


@composer
def average(*frames: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.concat(frames)
        .group_by("time", "asset_id")
        .agg(pl.col("value").mean().alias("value"))
        .sort("time", "asset_id")
    )
```

## 因果 Prediction Processing

`smooth_prediction(prediction, config, evaluation_calendar=..., state=None)`
只接受 `PredictionPanel`，返回包含 `.prediction` 与 `.state` 的
`PredictionSmoothingResult`。配置分别为 `PredictionSmoothingConfig(method="none")`、
`PredictionSmoothingConfig(method="sma", window=3)`、
`PredictionSmoothingConfig(method="ewma", half_life=2)`。

日历必须严格递增且唯一，窗口单位为评估期（日频交易日、月频评估点），不是自然日。
SMA 要求完整连续有限值窗口；EWMA 从首个有限值初始化，递推系数为
`1 - 2**(-1/half_life)`。缺失、非有限值与退出 Universe 均逐资产清空历史，不补零也不延用。

输入 Domain 必须覆盖完整连续日历区间。首次计算需从日历首日开始；断点续算必须传入
紧邻区间前一评估点的匹配 checkpoint。通过 `state.to_dict()` 与
`PredictionSmoothingState.from_dict(...)` 显式持久化；配置或历史日历改变会拒绝恢复。
UI 日期筛选不能重启平滑。该数值函数不是新增的无限制图/DSL 操作。

## 边界

`WeightedRegressionMoments.scale_weights(factor)` 用正的有限标量缩放所有已累计的
加权统计量，并保留 `observation_count`。它只修改当前累积器，先前 `copy()` 的快照
保持独立。流式衰减可以先缩放旧统计量再加入下一批，以最新已观测批次作为参考，
不依赖未来数据终点。衰减政策与训练日期选择仍由 Workbench 管理。

公开 API 面向 Polars、`Panel` 和 `Graph`。`bagelquant-core` 不负责数据获取、凭证管理、持久化、组合模拟或应用 UI。

