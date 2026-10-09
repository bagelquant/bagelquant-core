"""Public typed graph and cache proofs with adversarial sparse input."""

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import CoreStore, Domain, ExecutionRuntime, Graph, Node
from bagelquant_core.operator import (
    add,
    identity,
    mul,
    negate,
    rolling_elastic_net_prediction,
)
from test_global_graph import finish, global_graph, source


@pytest.mark.parametrize("value_type", ["numeric", "prediction", "weights", "category"])
@pytest.mark.parametrize("empty", [False, True])
def test_identity_preserves_type_sparse_coordinates_and_traces(value_type, empty):
    day = date(2024, 1, 2)
    frame = pl.DataFrame(
        {
            "time": [day] * 3,
            "asset_id": ["a", "b", "c"],
            "value": [0.0, float("nan"), None],
            "available": [day, None, day],
        }
    )
    if value_type == "category":
        frame = frame.with_columns(pl.Series("value", ["zero", None, "one"]))
    if empty:
        frame = frame.head(0)
    x = Node.from_domain(
        frame,
        Domain(calendar=[day], universe=["a", "b", "c", "missing"]),
        value_type=value_type,
        source_key="x",
        trace_columns=("available",),
    )
    out = identity(x)
    assert out.logical_id == x.logical_id
    assert out.value_type == value_type
    assert_frame_equal(
        out.collect(dense=False, include_traces=True),
        x.collect(dense=False, include_traces=True),
        check_exact=True,
    )


@pytest.mark.parametrize("operation", [add, mul])
@pytest.mark.parametrize("value_type", ["numeric", "prediction", "weights"])
def test_binary_exchange_uses_same_actual_definition_with_nan_sparse_traces(
    operation, value_type
):
    day = date(2024, 1, 2)
    domain = Domain(calendar=[day], universe=["a", "b", "c"])

    def node(values, key):
        return Node.from_domain(
            pl.DataFrame(
                {
                    "time": [day] * len(values),
                    "asset_id": ["a", "b", "c"][: len(values)],
                    "value": values,
                    "available": [day] * len(values),
                }
            ),
            domain,
            source_key=key,
            value_type=value_type,
            trace_columns=("available",),
        )

    a, b = node([0.0, float("nan"), 4.0], "a"), node([-0.0, 2.0], "b")
    left, right = operation(a, b), operation(b, a)
    assert left.logical_id == right.logical_id
    assert left.definition().inputs == right.definition().inputs
    assert_frame_equal(
        left.compute().collect(include_traces=True),
        right.compute().collect(include_traces=True),
        check_exact=True,
    )
    with pytest.raises(TypeError):
        operation(a, Graph(b))


def test_training_output_type_survives_dsl_serialization_and_cache(tmp_path):
    x = source(5)
    trained = rolling_elastic_net_prediction(
        x, labels=x, min_samples=100, max_samples=100
    )
    assert trained.value_type == "prediction"
    with pytest.raises(TypeError):
        add(trained, x)
    parsed = Graph.from_dsl(
        "output = rolling_elastic_net_prediction(x, labels=x, min_samples=100, max_samples=100)",
        inputs={"x": x},
    )
    assert parsed._single_output().value_type == "prediction"
    restored = Graph.from_logical_spec(parsed.logical_spec(), inputs={"x": x})
    assert restored._single_output().value_type == "prediction"
    graph, store, root = global_graph(tmp_path, parsed)
    receipt = finish(
        graph.plan_update({"main": {"x": x}}, through=x.domain.times.max())
    )
    assert store.read(receipt["results"]["main"][root]).panel.value_type == "prediction"


def test_retained_binding_cannot_upgrade_transient_source_evidence(tmp_path):
    x = source(3)
    a = negate(x)
    spec = negate(a).graph.logical_spec()
    transient = Node.from_domain(x.lazy(), x.domain)
    bound = Graph.from_logical_spec(
        spec, inputs={"x": x}, node_bindings={a.logical_id: transient}
    )
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "values")
    with pytest.raises(ValueError, match="evidence"):
        bound.compute(runtime=ExecutionRuntime(materialization_store=store))
    assert not store.inventory()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_optional_damaged_checkpoint_falls_back_but_history_stays_invalid(
    tmp_path, damage
):
    from bagelquant_core.operator import ewm_mean

    first = source(35, dynamic=True)
    graph, store, root = global_graph(
        tmp_path / "incremental", ewm_mean(first, alpha=0.4)
    )
    old = finish(
        graph.plan_update({"main": {"x": first}}, through=first.domain.times.max())
    )
    old_identity = old["results"]["main"][root]
    file = store._file(store.manifest(old_identity)["partitions"][0]["$file"])
    if damage == "missing":
        file.unlink()
    else:
        file.write_bytes(b"corrupt optional candidate")
    x = source(45, dynamic=True)
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max())
    result = finish(plan)
    cold, fresh, cold_root = global_graph(tmp_path / "cold", ewm_mean(x, alpha=0.4))
    expected = finish(
        cold.plan_update({"main": {"x": x}}, through=x.domain.times.max())
    )
    assert_frame_equal(
        store.read(result["results"]["main"][root]).panel.collect(include_traces=True),
        fresh.read(expected["results"]["main"][cold_root]).panel.collect(
            include_traces=True
        ),
        check_exact=True,
    )
    assert plan.resource_usage["main"][root]["checkpoint_through"] is None
    with pytest.raises((ValueError, OSError)):
        store.verify_identity(old_identity)
