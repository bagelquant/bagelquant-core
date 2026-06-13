# Quick Start

`bagelquant-core` is the panel and graph foundation for BagelQuant research.
Use it when raw research inputs are already available as long-form Polars data
and you want reproducible factor logic.

## Install

```bash
uv add bagelquant-core
```

For local development from this repository:

```bash
uv run python example.py
```

## Build A Domain

A `Domain` defines the trading sessions and asset universe used by every input
panel. The package does not download calendars or security masters; callers
provide them from a data layer.

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

## Create Panels

`Panel.from_domain` aligns raw frames to the domain. Public panel data is
long-form with `time`, `asset_id`, and `value` columns.

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

## Compose A Factor Graph

Transformers are unary operations. Composers combine one or more inputs. Both
return lazy `Graph` objects.

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

## Execute

Call `compute()` on the downstream graph. The execution runtime evaluates
upstream dependencies once and caches intermediate panels for the current run.

```python
signal.compute()
result = signal.output
frame = result.data
```

Use `frame` as input to downstream portfolio construction or backtesting.

