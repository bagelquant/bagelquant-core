import pandas as pd

from bagelquant_core import Graph, Panel
from bagelquant_core.execution import _ExecutionRuntime
from bagelquant_core.transformer import rank, zscore


def test_execution_cache_reuse() -> None:
    price = Panel(pd.DataFrame({"a": [1, 2]}), name="price")
    ranked = rank(price, name="ranked")

    runtime = _ExecutionRuntime()
    runtime.run(ranked)
    initial_cache_size = len(runtime.cache)

    runtime.run(ranked)
    assert len(runtime.cache) == initial_cache_size


def test_execution_cache_changes_on_new_inputs() -> None:
    runtime = _ExecutionRuntime()

    price_a = Panel(pd.DataFrame({"a": [1, 2]}), name="price_a")
    runtime.run(rank(price_a, name="ranked_a"))
    cache_size = len(runtime.cache)

    price_b = Panel(pd.DataFrame({"a": [10, 20]}), name="price_b")
    runtime.run(rank(price_b, name="ranked_b"))

    assert len(runtime.cache) > cache_size


def test_multi_output_graph_returns_named_panels() -> None:
    price = Panel(pd.DataFrame({"a": [1, 2]}), name="price")
    outputs = Graph(
        outputs=[rank(price, name="ranked"), zscore(price, name="zscore")]
    )

    results = outputs.compute()

    assert set(results) == {"ranked", "zscore"}
    assert all(isinstance(panel, Panel) for panel in results.values())
