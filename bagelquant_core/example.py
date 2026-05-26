import numpy as np
import pandas as pd

from panel import Panel
from transformer import Rank, ZScore, Winsorize
from composer import Div, WeightedSum
from execution import ExecutionEngine


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

bm = Div(
    book,
    price,
)

bm_processed = Rank(
    ZScore(
        Winsorize(
            bm
        )
    )
)


# -------------------------
# Quality Factor (Operator chain)
# -------------------------

quality_processed = Rank(
    ZScore(
        quality
    )
)


# -------------------------
# Final Prediction (Composer)
# -------------------------

prediction = WeightedSum(
    bm_processed,
    quality_processed,
    weights=[0.5, 0.5],
)


# =========================================================
# 4. Downstream usage
# =========================================================

signal = Rank(prediction)
normalized_signal = ZScore(prediction)


# =========================================================
# 5. Execution
# =========================================================

engine = ExecutionEngine()


print("\n====================")
print("RUN: SIGNAL")
print("====================")

signal_result = engine.run(signal)
print(signal_result)


print("\n====================")
print("RUN: NORMALIZED")
print("====================")

normalized_result = engine.run(normalized_signal)
print(normalized_result)


# =========================================================
# 6. Cache demonstration (key part)
# =========================================================

print("\n====================")
print("CACHE RE-RUN SIGNAL")
print("====================")

# rerun same node should trigger cache hits
signal_result_2 = engine.run(signal)
print(signal_result_2)

