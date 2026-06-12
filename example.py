from __future__ import annotations

import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import div, weighted_mean
from bagelquant_core.transformer import rank, rolling_mean, winsorize, zscore


def main() -> None:
    domain = Domain(
        calendar=["2024-01-02", "2024-01-03", "2024-01-04"],
        universe=["AAA", "BBB"],
    )
    price = Panel.from_domain(
        pl.DataFrame(
            {
                "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
                "asset_id": ["AAA"] * 3 + ["BBB"] * 3,
                "value": [100.0, 102.0, 101.0, 50.0, 51.0, 53.0],
            }
        ),
        domain,
        name="price",
    )
    book = Panel.from_domain(
        pl.DataFrame(
            {
                "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
                "asset_id": ["AAA"] * 3 + ["BBB"] * 3,
                "value": [40.0, 41.0, 42.0, 20.0, 21.0, 22.0],
            }
        ),
        domain,
        name="book",
    )

    bm_ratio = div(book, price, name="bm_ratio")
    value_factor = rank(zscore(winsorize(bm_ratio)), name="value_factor")
    signal = rolling_mean(
        weighted_mean(value_factor, value_factor, weights=[0.5, 0.5]),
        window=2,
        min_periods=1,
        name="signal",
    )
    signal.compute()

    print(signal.output.data.to_dicts())


if __name__ == "__main__":
    main()
