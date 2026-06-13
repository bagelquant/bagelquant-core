# 快速开始

`bagelquant-core` 是 BagelQuant 的面板数据与图计算基础。当研究数据已经是 long-form Polars 数据，并且你希望用可复现的方式组织因子逻辑时，可以使用它。

## 安装

```bash
uv add bagelquant-core
```

本地开发时可以从仓库根目录运行：

```bash
uv run python example.py
```

## 构建 Domain

`Domain` 定义交易日历和资产池。包本身不会下载交易日历或证券主数据，这些信息应由数据层提供。

```python
import polars as pl

from bagelquant_core import Domain

domain = Domain(
    calendar=pl.date_range(
        pl.date(2024, 1, 1),
        pl.date(2024, 12, 31),
        interval="1d",
        eager=True,
    ),
    universe=["AAPL", "MSFT"],
)
```

## 创建 Panel

`Panel.from_domain` 会把原始数据对齐到研究域。公开面板数据统一使用 `time`、`asset_id` 和 `value` 列。

```python
from bagelquant_core import Panel

price = Panel.from_domain(
    pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "asset_id": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "value": [185.0, 370.0, 187.0, 372.0],
        }
    ),
    domain,
    name="price",
)
book = Panel.from_domain(book_df, domain, name="book")
quality = Panel.from_domain(quality_df, domain, name="quality")
```

## 组合因子图

Transformer 是一元操作，composer 用于组合一个或多个输入。二者都会返回惰性的 `Graph`。

```python
from bagelquant_core.composer import div, weighted_sum
from bagelquant_core.transformer import rank, rolling_mean, winsorize, zscore

bm_ratio = div(book, price, name="bm_ratio")
bm_factor = rank(zscore(winsorize(bm_ratio)), name="bm_factor")
quality_factor = rank(zscore(quality), name="quality_factor")

prediction = weighted_sum(
    bm_factor,
    quality_factor,
    weights=[0.5, 0.5],
    name="prediction",
)

signal = rolling_mean(rank(prediction), window=20, name="signal")
```

## 执行

在最下游图上调用 `compute()`。执行运行时会计算上游依赖，并在本次执行中缓存中间结果。

```python
signal.compute()
result = signal.output
frame = result.data
```

`frame` 可以继续交给组合构建或回测模块。

