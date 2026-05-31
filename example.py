from __future__ import annotations

import numpy as np
import pandas as pd

from bagelquant_core import ExecutionEngine, Graph, Panel
from bagelquant_core.composer import div, weighted_sum
from bagelquant_core.transformer import rank, winsorize, zscore


def main() -> None:
    dates = pd.date_range("2024-01-01", periods=100)
    stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "DIS"]

    price = Panel(
        pd.DataFrame(np.random.rand(100, 10) * 100, index=dates, columns=stocks),
        name="price",
    )
    book = Panel(
        pd.DataFrame(np.random.rand(100, 10) * 50, index=dates, columns=stocks),
        name="book",
    )
    quality = Panel(
        pd.DataFrame(np.random.rand(100, 10), index=dates, columns=stocks),
        name="quality",
    )

    bm_ratio = div(book, price, name="bm_ratio")
    bm_factor = rank(zscore(winsorize(bm_ratio)), name="bm_factor")
    quality_factor = rank(zscore(quality), name="quality_factor")

    prediction = weighted_sum(
        bm_factor,
        quality_factor,
        weights=[0.5, 0.5],
        name="prediction",
    )
    signal = rank(prediction, name="signal")
    normalized_signal = zscore(prediction, name="normalized_signal")

    strategy = Graph(outputs=[signal, normalized_signal])
    engine = ExecutionEngine()
    strategy.compute(engine)

    print(signal.output.data)
    print(normalized_signal.output.data)

    # Intermediate graph outputs are populated by downstream execution.
    print(prediction.output.data)


if __name__ == "__main__":
    main()
