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

## 边界

公开 API 面向 Polars、`Panel` 和 `Graph`。`bagelquant-core` 不负责数据获取、凭证管理、持久化、组合模拟或应用 UI。

