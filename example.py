from __future__ import annotations

import numpy as np
import pandas as pd

from bagelquant_core import Graph, Panel
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


def main() -> None:
    rng = np.random.default_rng(seed=7)
    dates = pd.date_range("2024-01-01", periods=120)
    stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "DIS"]

    simulated_log_returns = rng.normal(loc=0.0005, scale=0.02, size=(120, 10))
    price = Panel(
        pd.DataFrame(
            100 * np.exp(simulated_log_returns.cumsum(axis=0)),
            index=dates,
            columns=stocks,
        ),
        name="price",
    )
    book = Panel(
        pd.DataFrame(rng.uniform(25, 75, size=(120, 10)), index=dates, columns=stocks),
        name="book",
    )
    quality = Panel(
        pd.DataFrame(rng.normal(size=(120, 10)), index=dates, columns=stocks),
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
    signal = zscore(rank(smoothed_prediction), name="signal")

    strategy = Graph(outputs=[signal, prediction, volatility])
    outputs = strategy.compute()

    print(f"Computed {len(strategy.nodes)} graph nodes")
    for name, output in outputs.items():
        print(f"\n{name}:")
        print(output.data.tail(3).round(4))


if __name__ == "__main__":
    main()
