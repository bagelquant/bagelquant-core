import pandas as pd

from bagelquant_core import ExecutionEngine, Panel, Rank


def test_execution_cache_reuse() -> None:
    data = pd.DataFrame({"a": [1, 2]})
    panel = Panel("price", data)
    node = Rank(panel, name="ranked")

    engine = ExecutionEngine()
    engine.run(node)
    initial_cache_size = len(engine.cache)

    engine.run(node)
    assert len(engine.cache) == initial_cache_size


def test_execution_cache_changes_on_new_inputs() -> None:
    data_a = pd.DataFrame({"a": [1, 2]})
    data_b = pd.DataFrame({"a": [10, 20]})

    engine = ExecutionEngine()

    panel_a = Panel("price", data_a)
    node_a = Rank(panel_a, name="ranked_a")
    engine.run(node_a)
    cache_size = len(engine.cache)

    panel_b = Panel("price", data_b)
    node_b = Rank(panel_b, name="ranked_b")
    engine.run(node_b)

    assert len(engine.cache) > cache_size
