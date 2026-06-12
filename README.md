# BagelQuant Core

`bagelquant-core` is the Polars-native panel and graph layer for BagelQuant.

Panel data is long-form and keyed by `time` and `asset_id`. Numeric panels use a
single `value` column:

```python
import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import div
from bagelquant_core.transformer import rank, zscore

domain = Domain(calendar=["2024-01-02", "2024-01-03"], universe=["AAA", "BBB"])
price = Panel.from_domain(
    pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02"],
            "asset_id": ["AAA", "BBB"],
            "value": [100.0, 50.0],
        }
    ),
    domain,
    name="price",
)
book = Panel.from_domain(
    pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02"],
            "asset_id": ["AAA", "BBB"],
            "value": [40.0, 25.0],
        }
    ),
    domain,
    name="book",
)

factor = rank(zscore(div(book, price)), name="factor")
factor.compute()

print(factor.output.data)
```

Time-series operations group by `asset_id` and order by `time`.
Cross-sectional operations group by `time`.
Composer operations join inputs on `(time, asset_id)`.

## Development

```bash
uv run ruff check .
uv run python -m pytest
```
