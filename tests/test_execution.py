import pandas as pd

from bagelquant_core import ExecutionEngine, Graph, Panel
from bagelquant_core.transformer import rank, zscore


def test_execution_cache_reuse() -> None:
    price = Panel(pd.DataFrame({"a": [1, 2]}), name="price")
    ranked = rank(price, name="ranked")

    engine = ExecutionEngine()
    ranked.compute(engine)
    initial_cache_size = len(engine.cache)

    ranked.compute(engine)
    assert len(engine.cache) == initial_cache_size


def test_execution_cache_changes_on_new_inputs() -> None:
    engine = ExecutionEngine()

    price_a = Panel(pd.DataFrame({"a": [1, 2]}), name="price_a")
    rank(price_a, name="ranked_a").compute(engine)
    cache_size = len(engine.cache)

    price_b = Panel(pd.DataFrame({"a": [10, 20]}), name="price_b")
    rank(price_b, name="ranked_b").compute(engine)

    assert len(engine.cache) > cache_size


def test_multi_output_graph_returns_named_panels() -> None:
    price = Panel(pd.DataFrame({"a": [1, 2]}), name="price")
    outputs = Graph(
        outputs=[rank(price, name="ranked"), zscore(price, name="zscore")]
    )

    results = outputs.compute()

    assert set(results) == {"ranked", "zscore"}
    assert all(isinstance(panel, Panel) for panel in results.values())
