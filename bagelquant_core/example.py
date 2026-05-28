from __future__ import annotations

import pathlib
import sys

if __package__ in (None, ""):
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from bagelquant_core import (
    Div,
    ExecutionEngine,
    Graph,
    Panel,
    Rank,
    WeightedSum,
    Winsorize,
    ZScore,
)


# =========================================================
# 1. Sample Data
# =========================================================

def main() -> None:
    # =========================================================
    # 1. Sample Data
    # =========================================================

    dates = pd.date_range(
        "2024-01-01",
        periods=100,
    )

    stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "DIS"]

    price_df = pd.DataFrame(
        np.random.rand(100, 10) * 100,
        index=dates,
        columns=stocks,
    )

    book_df = pd.DataFrame(
        np.random.rand(100, 10) * 50,
        index=dates,
        columns=stocks,
    )

    quality_df = pd.DataFrame(
        np.random.rand(100, 10),
        index=dates,
        columns=stocks,
    )

    # =========================================================
    # 2. Panels (0 → 1)
    # =========================================================

    price = Panel("price", price_df)
    book = Panel("book", book_df)
    quality = Panel("quality", quality_df)

    # =========================================================
    # 3. Build Factor DAG
    # =========================================================

    # -------------------------
    # BM Factor (Composer chain)
    # -------------------------

    bm = Div(book, price, name="bm_ratio")

    bm_processed = Rank(
        ZScore(
            Winsorize(bm),
        ),
        name="bm_factor",
    )

    # -------------------------
    # Quality Factor (Operator chain)
    # -------------------------

    quality_processed = Rank(ZScore(quality), name="quality_factor")

    # -------------------------
    # Final Prediction (Composer)
    # -------------------------

    prediction = WeightedSum(
        bm_processed,
        quality_processed,
        weights=[0.5, 0.5],
        name="prediction",
    )

    # =========================================================
    # 4. Downstream usage
    # =========================================================

    signal = Rank(prediction, name="signal")
    normalized_signal = ZScore(prediction, name="normalized_signal")

    # =========================================================
    # 5. Execution
    # =========================================================

    engine = ExecutionEngine()
    graph = Graph(outputs=[signal, normalized_signal])

    print("\n====================")
    print("RUN: GRAPH")
    print("====================")

    results = engine.run(graph)
    print(results[signal.name])
    print(results[normalized_signal.name])

    # =========================================================
    # 6. Cache demonstration (key part)
    # =========================================================

    print("\n====================")
    print("CACHE RE-RUN SIGNAL")
    print("====================")

    # rerun same node should trigger cache hits
    signal_result_2 = engine.run(signal)
    print(signal_result_2)


if __name__ == "__main__":
    main()
