from __future__ import annotations

import numpy as np
import pandas as pd

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import (
    div,
    maximum,
    mean,
    minimum,
    product,
    weighted_mean,
)
from bagelquant_core.transformer import (
    diff,
    lag,
    log,
    negate,
    pct_change,
    rank,
    rolling_mean,
    rolling_std,
    signed_log1p,
    winsorize,
    zscore,
)
from bagelquant_core.transformer import transformer


# User custom transformer to implement a long-short equal weight strategy.
@transformer
def long_short_equal_weight(
    frame: pd.DataFrame,
    *,
    count: int,
) -> pd.DataFrame:
    """Long the top assets and short the bottom assets with equal leg weights."""

    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("long_short_equal_weight count must be a positive integer")

    ranks = frame.rank(axis=1, method="first", ascending=False)
    valid_count = frame.notna().sum(axis=1)
    enough_assets = valid_count >= count * 2
    long_mask = ranks.le(count).mul(enough_assets, axis=0)
    short_mask = ranks.gt(valid_count.sub(count), axis=0).mul(enough_assets, axis=0)
    return long_mask.astype(float).sub(short_mask.astype(float)).div(count)


def main() -> None:
    rng = np.random.default_rng(seed=7)
    stocks = [f"STOCK_{stock_id:04d}" for stock_id in range(1, 501)]
    domain = Domain(
        calendar=pd.bdate_range("2015-01-01", "2024-12-31"),
        universe=stocks,
    )
    dates = domain.sessions
    rows = len(dates)
    columns = len(stocks)

    simulated_log_returns = rng.normal(loc=0.0005, scale=0.02, size=(rows, columns))
    price = Panel.from_domain(
        pd.DataFrame(
            100 * np.exp(simulated_log_returns.cumsum(axis=0)),
            index=dates,
            columns=stocks,
        ),
        domain,
        name="price",
    )
    book = Panel.from_domain(
        pd.DataFrame(rng.uniform(25, 75, size=(rows, columns)), index=dates, columns=stocks),
        domain,
        name="book",
    )
    quality = Panel.from_domain(
        pd.DataFrame(rng.normal(size=(rows, columns)), index=dates, columns=stocks),
        domain,
        name="quality",
    )

    # Fundamental branch: combine value and quality while clipping outliers.
    bm_ratio = div(book, price, name="bm_ratio")
    value_factor = rank(
        zscore(winsorize(bm_ratio, lower=0.05, upper=0.95)),
        name="value_factor",
    )
    quality_factor = rank(zscore(winsorize(quality)), name="quality_factor")
    fundamental_factor = weighted_mean(
        value_factor,
        quality_factor,
        weights=[0.6, 0.4],
        name="fundamental_factor",
    )

    # Technical branch: reward medium-term momentum and penalize volatility.
    daily_return = pct_change(price, name="daily_return")
    log_price = log(price, name="log_price")
    momentum = diff(log_price, periods=20, name="momentum")
    momentum_factor = rank(zscore(winsorize(momentum)), name="momentum_factor")
    volatility = rolling_std(
        daily_return,
        window=20,
        min_periods=10,
        name="volatility",
    )
    low_vol_factor = rank(negate(zscore(volatility)), name="low_vol_factor")
    technical_factor = mean(
        momentum_factor,
        low_vol_factor,
        name="technical_factor",
    )

    # Composite branch: include a nonlinear interaction and a conservative
    # agreement score between the two major branches.
    interaction = signed_log1p(
        product(fundamental_factor, technical_factor),
        name="interaction",
    )
    agreement_floor = minimum(
        fundamental_factor,
        technical_factor,
        name="agreement_floor",
    )
    agreement_ceiling = maximum(
        fundamental_factor,
        technical_factor,
        name="agreement_ceiling",
    )
    agreement = mean(agreement_floor, agreement_ceiling, name="agreement")
    prediction = weighted_mean(
        fundamental_factor,
        technical_factor,
        interaction,
        agreement,
        weights=[0.35, 0.35, 0.15, 0.15],
        name="prediction",
    )

    smoothed_prediction = rolling_mean(
        prediction,
        window=5,
        min_periods=1,
        name="smoothed_prediction",
    )
    signal = long_short_equal_weight(
        smoothed_prediction,
        count=20,
        name="signal",
    )
    pnl = product(
        lag(signal, name="lagged_signal"),
        daily_return,
        name="pnl",
    )
    pnl.compute()
    daily_pnl = pnl.output.data.sum(axis=1)

    print(f"Computed {len(pnl.nodes)} graph nodes")
    print(f"Dataset shape: {rows} trading sessions x {columns} stocks")
    print("\nsignal:")
    print(signal.output.data.tail(3).round(4))
    print("\ndaily pnl:")
    print(daily_pnl.tail(3).round(6))
    print(f"\ncumulative pnl: {daily_pnl.sum():.6f}")


if __name__ == "__main__":
    from time import perf_counter

    start_time = perf_counter()
    main()
    time_taken = perf_counter() - start_time
    print(f"\nExecution time: {time_taken:.2f} seconds \nor {time_taken/60:.2f} minutes")
