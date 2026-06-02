import pandas as pd

from bagelquant_core import Domain, Graph, Panel
from bagelquant_core.composer import add
from bagelquant_core.execution import _ExecutionRuntime
from bagelquant_core.transformer import rank, zscore
from tests.helpers import make_panel


def test_execution_cache_reuse() -> None:
    price = make_panel(pd.DataFrame({"a": [1, 2]}), name="price")
    ranked = rank(price, name="ranked")

    runtime = _ExecutionRuntime()
    runtime.run(ranked)
    initial_cache_size = len(runtime.cache)

    runtime.run(ranked)
    assert len(runtime.cache) == initial_cache_size


def test_execution_cache_changes_on_new_inputs() -> None:
    runtime = _ExecutionRuntime()

    price_a = make_panel(pd.DataFrame({"a": [1, 2]}), name="price_a")
    runtime.run(rank(price_a, name="ranked_a"))
    cache_size = len(runtime.cache)

    price_b = make_panel(pd.DataFrame({"a": [10, 20]}), name="price_b")
    runtime.run(rank(price_b, name="ranked_b"))

    assert len(runtime.cache) > cache_size


def test_multi_output_graph_returns_named_panels() -> None:
    price = make_panel(pd.DataFrame({"a": [1, 2]}), name="price")
    outputs = Graph(
        outputs=[rank(price, name="ranked"), zscore(price, name="zscore")]
    )

    results = outputs.compute()

    assert set(results) == {"ranked", "zscore"}
    assert all(isinstance(panel, Panel) for panel in results.values())


def test_execution_reuses_hashes_for_already_aligned_inputs(monkeypatch) -> None:
    def fail_on_hash(_: pd.DataFrame) -> str:
        raise AssertionError("execution should reuse stored panel hashes")

    monkeypatch.setattr("bagelquant_core.execution.hash_dataframe", fail_on_hash)

    left = make_panel(pd.DataFrame({"a": [1, 2]}), name="left")
    right = make_panel(pd.DataFrame({"a": [10, 20]}), name="right")

    assert add(left, right).compute().data["a"].tolist() == [11, 22]


def test_execution_cache_distinguishes_domain_signatures() -> None:
    runtime = _ExecutionRuntime()
    us = Domain(region="US", universe=["a"], start_date="2024-01-02", end_date="2024-01-03")
    hk = Domain(region="HK", universe=["a"], start_date="2024-01-02", end_date="2024-01-03")
    frame = pd.DataFrame({"a": [1, 2]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    runtime.run(rank(Panel.from_domain(frame, us), name="ranked"))
    runtime.run(rank(Panel.from_domain(frame, hk), name="ranked"))

    assert len(runtime.cache) == 2
