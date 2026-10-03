from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import (
    CategoryPanel, Domain, ExecutionRuntime, Graph, LogicalGraphSpec, LogicalNodeSpec,
    MaterializationLookup, MaterializationStatus, Panel,
    PredictionPanel, canonicalize_graph, canonicalize_values, capture_operator_checkpoints,
    capture_training_audits, project_domain, rebalance, rolling_elastic_net_prediction,
)
from bagelquant_core import OPERATOR_REGISTRY
from bagelquant_core.operation_contract import ExecutionMode, OperationContract, TraceRule
from bagelquant_core.prediction import IdentityPredictionComposer
from bagelquant_core.resources import ResourceLimits, resource_limits
from bagelquant_core.transformer import negate, rank, rolling_mean
from bagelquant_core.transformer.core import transformer
from bagelquant_core.composer import add


@pytest.fixture(autouse=True)
def isolate_extension_catalog(monkeypatch):
    monkeypatch.setattr(OPERATOR_REGISTRY, "_items", dict(OPERATOR_REGISTRY._items))


class FrameStore:
    """A fake adapter whose primary values survive fresh Runtime instances."""

    def __init__(self, root, records=None):
        self.root = root
        self.records = dict(records or {})
        self.published = []
        self.partial = None

    def query(self, key):
        record = self.records.get(key.identity)
        if record is not None:
            return MaterializationLookup(MaterializationStatus.HIT, record)
        if self.partial is not None:
            return MaterializationLookup(MaterializationStatus.PARTIAL, self.partial, "unverified prefix")
        return MaterializationLookup(MaterializationStatus.MISS)

    def publish(self, record):
        path = self.root / f"{record.key.identity}.parquet"
        record.panel.collect(dense=False, include_traces=True).write_parquet(path)
        panel = type(record.panel).from_domain(pl.scan_parquet(path), record.panel.domain,
            identity=record.key.identity, trace_identity=record.panel.trace_identity,
            trace_columns=record.panel.trace_columns)
        self.records[record.key.identity] = replace(record, panel=panel,
            artifacts=deepcopy(record.artifacts), checkpoint=deepcopy(record.checkpoint),
            training_audits=deepcopy(record.training_audits))
        self.published.append(record.key)


def source(*, name="close", receipt="close.v1", traced=False):
    days = [date(2024, 1, day) for day in (2, 3, 4)]
    domain = Domain(calendar=days, universe=["A", "B"])
    frame = pl.DataFrame({"time": [day for day in days for _ in range(2)],
                          "asset_id": ["A", "B"] * 3,
                          "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    if traced:
        frame = frame.with_columns(pl.col("time").alias("available_date"))
    return Panel.from_domain(frame, domain, name=name, source_key="feed.close", identity=receipt,
        trace_columns=("available_date",) if traced else (), trace_identity="availability.v1" if traced else None)


def test_bound_input_keeps_the_complete_immutable_logical_contract():
    input_node = LogicalNodeSpec.create(node_type="input", input_key="feed.close",
        parameters={"value_type":"panel", "contract":{"unit":"dimensionless", "column":"value"}})
    specification = LogicalGraphSpec({"input":input_node.node_id}, (input_node,))
    graph = Graph.from_logical_spec(specification, inputs={"feed.close":source()})
    assert graph.nodes[0].logical_id == input_node.node_id
    assert graph.logical_spec().to_dict() == specification.to_dict()
    assert graph.compute().collect().equals(source().collect())


def test_logical_identity_ignores_labels_metadata_receipts_and_domains():
    first = rolling_mean(source(name="input_a"), window=2, name="first", metadata={"stage": "research"})
    second = rolling_mean(source(name="input_b", receipt="close.v2"), window=2, min_periods=None,
                          name="second", metadata={"stage": "fail"})
    assert first._single_output().logical_id == second._single_output().logical_id
    union = first.logical_spec().union(second.logical_spec())
    assert len(union.nodes) == 2
    assert len(set(union.outputs.values())) == 1
    assert LogicalGraphSpec.from_dict(union.to_dict()).identity == union.identity
    narrow = Panel.from_domain(source().collect().filter(pl.col("asset_id") == "A"),
        Domain(calendar=source().domain.times, universe=["A"]), source_key="feed.close")
    assert rolling_mean(narrow, window=2)._single_output().logical_id == first._single_output().logical_id
    assert rolling_mean(narrow, window=3)._single_output().logical_id != first._single_output().logical_id
    with pytest.raises(TypeError):
        union.nodes[-1].parameters["window"] = 99


def test_logical_serialization_rejects_tampering_and_keeps_dependency_roles():
    first = source()
    second = Panel.from_domain(first.collect(), first.domain, name="other", source_key="feed.other")
    @transformer
    def auxiliary(frame: pl.DataFrame, *, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        return frame
    a = auxiliary(first, left=first, right=second)
    b = auxiliary(first, left=second, right=first)
    assert a._single_output().logical_id != b._single_output().logical_id
    raw = a.logical_spec().to_dict()
    raw["nodes"][-1]["parameters"]["tampered"] = True
    with pytest.raises(ValueError, match="content"):
        LogicalGraphSpec.from_dict(raw)
    with pytest.raises(TypeError, match="unsupported logical"):
        from bagelquant_core.logical import canonical_parameters
        canonical_parameters(object())


def test_logical_union_binds_aliases_and_prunes_presentation_provenance():
    first = rolling_mean(source(), window=2, name="one")
    second = negate(rolling_mean(source(), window=2, name="two"), name="negative")
    union = first.logical_spec().union(second.logical_spec())
    rebound = Graph.from_logical_spec(union, inputs={"feed.close": source(name="renamed")})
    assert len(rebound.nodes) == 3
    results = rebound.compute(dense_output=False)
    assert set(results) == {"one", "negative"}
    assert_frame_equal(results["negative"].collect(), results["one"].collect().with_columns(-pl.col("value")))
    template = first.spec().to_dict()
    template["nodes"].insert(0, {"name": "raw-provenance", "node_type": "panel", "inputs": [], "config": {}})
    template["nodes"][-1]["metadata"]["context_dependencies"] = {"provenance": ["raw-provenance"]}
    assert len(canonicalize_graph(template).nodes) == 2


def test_sequential_variants_reuse_persisted_intermediate_after_runtime_restart(tmp_path):
    calls = []
    @transformer
    def counted(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("compute")
        return frame.with_columns((pl.col("value") * 2).alias("value"))
    store = FrameStore(tmp_path)
    first = rolling_mean(counted(source()), window=2, name="first")
    first.compute(runtime=ExecutionRuntime(materialization_store=store), dense_output=False)
    restarted = FrameStore(tmp_path, store.records)
    second = negate(counted(source(name="fresh")), name="second")
    runtime = ExecutionRuntime(materialization_store=restarted)
    result = second.compute(runtime=runtime, dense_output=False)
    assert calls == ["compute"]
    assert runtime.persistent_hits == 1
    assert len(restarted.published) == 1
    assert result.collect()["value"].to_list() == [-2, -4, -6, -8, -10, -12]


def test_batch_independent_equal_roots_execute_once_and_match_sequential(tmp_path):
    calls = []
    @transformer
    def counted(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(1)
        return frame
    a = rolling_mean(counted(source()), window=2, name="a")
    b = negate(counted(source()), name="b")
    union = a.logical_spec().union(b.logical_spec())
    store = FrameStore(tmp_path)
    results = Graph.from_logical_spec(union, inputs={"feed.close": source()}).compute(
        runtime=ExecutionRuntime(materialization_store=store), dense_output=False)
    assert calls == [1]
    assert len(store.published) == 3
    restarted = FrameStore(tmp_path, store.records)
    runtime = ExecutionRuntime(materialization_store=restarted)
    sequential = Graph.from_logical_spec(b.logical_spec(), inputs={"feed.close": source()}).compute(runtime=runtime)
    assert_frame_equal(sequential.collect(), results["b"].collect())
    assert calls == [1]
    assert runtime.persistent_hits == 2


def test_changed_input_version_trace_or_implementation_never_hits(tmp_path, monkeypatch):
    calls = []
    @transformer(contract=OperationContract(execution=ExecutionMode.EAGER_BARRIER, trace_rule=TraceRule.PASSTHROUGH))
    def counted(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(1)
        return frame
    store = FrameStore(tmp_path)
    first = counted(source(traced=True))
    logical = first._single_output().logical_id
    first.compute(runtime=ExecutionRuntime(materialization_store=store))
    counted(source(receipt="close.v2", traced=True)).compute(runtime=ExecutionRuntime(materialization_store=store))
    changed_trace = source(traced=True)
    changed_trace._trace_identity = "availability.v2"
    counted(changed_trace).compute(runtime=ExecutionRuntime(materialization_store=store))
    monkeypatch.setattr(counted, "version", "2")
    assert counted(source(traced=True))._single_output().logical_id == logical
    counted(source(traced=True)).compute(runtime=ExecutionRuntime(materialization_store=store))
    assert calls == [1, 1, 1, 1]


def test_partial_candidate_is_not_silently_admitted(tmp_path):
    store = FrameStore(tmp_path)
    rolling_mean(source(), window=2).compute(runtime=ExecutionRuntime(materialization_store=store))
    store.partial = next(iter(store.records.values()))
    runtime = ExecutionRuntime(materialization_store=store)
    output = rolling_mean(source(receipt="revision.v2"), window=2).compute(runtime=runtime)
    assert runtime.persistent_partials == 1
    assert runtime.persistent_hits == 0
    assert output.collect()["value"].to_list() == [None, None, 2, 3, 4, 5]


def test_cache_preserves_prediction_category_and_availability_traces(tmp_path):
    store = FrameStore(tmp_path)
    prediction = IdentityPredictionComposer().compose(source(traced=True))
    first = prediction.compute(runtime=ExecutionRuntime(materialization_store=store), dense_output=False)
    second = prediction.compute(runtime=ExecutionRuntime(materialization_store=store), dense_output=False)
    assert isinstance(second, PredictionPanel)
    assert_frame_equal(first.collect(include_traces=True), second.collect(include_traces=True))
    category = CategoryPanel.from_domain(source().collect().with_columns(pl.lit("industry").alias("value")),
        source().domain, identity="industry.v1", source_key="industry")
    from bagelquant_core.transformer import identity
    identity(category).compute(runtime=ExecutionRuntime(materialization_store=store))
    assert isinstance(identity(category).compute(runtime=ExecutionRuntime(materialization_store=store)), CategoryPanel)


@pytest.mark.parametrize("operation", ["lag", "fillna_zero", "scalar", "implicit_composer"])
def test_lowered_prediction_plans_remain_typed_across_persistent_hits(tmp_path, operation):
    from bagelquant_core.composer import add, mean
    from bagelquant_core.transformer import constant, fillna_zero, lag
    prediction = IdentityPredictionComposer().compose(source(traced=True))
    shifted = lag(prediction, periods=1)
    graph = {"lag": shifted, "fillna_zero": fillna_zero(shifted),
             "scalar": add(prediction, constant(prediction, value=2)),
             "implicit_composer": mean(fillna_zero(shifted), prediction)}[operation]
    store = FrameStore(tmp_path)
    first = graph.compute(runtime=ExecutionRuntime(materialization_store=store), dense_output=False)
    runtime = ExecutionRuntime(materialization_store=FrameStore(tmp_path, store.records))
    second = graph.compute(runtime=runtime, dense_output=False)
    assert isinstance(first, PredictionPanel) and isinstance(second, PredictionPanel)
    assert runtime.persistent_misses == 0
    assert_frame_equal(first.collect(include_traces=True), second.collect(include_traces=True))


def test_projection_preserves_source_universe_rank_and_typed_traces(tmp_path):
    full = source(traced=True)
    membership = Panel.from_domain(full.collect().filter(pl.col("asset_id") == "A").with_columns(pl.lit(1.).alias("value")),
        Domain(calendar=full.domain.times, universe=["A"]), identity="membership.v1", source_key="member.rule")
    graph = project_domain(rank(full), membership=membership)
    store = FrameStore(tmp_path)
    output = graph.compute(runtime=ExecutionRuntime(materialization_store=store))
    assert output.domain.equivalent_to(membership.domain)
    assert output.collect()["value"].to_list() == [0.5, 0.5, 0.5]
    cached = graph.compute(runtime=ExecutionRuntime(materialization_store=store))
    assert_frame_equal(output.collect(include_traces=True), cached.collect(include_traces=True))


def test_explicit_rounding_and_bound_context_do_not_change_logical_node(tmp_path):
    rounded = canonicalize_values(source(), significant_digits=3)
    assert rounded.compute().collect()["value"].to_list() == [1., 2., 3., 4., 5., 6.]
    input_ = source()
    calendar = Panel.from_domain(input_.collect().with_columns(pl.lit(1.).alias("value")), input_.domain,
                                identity="calendar.v1", source_key="calendar")
    template = rebalance(input_, calendar=calendar, data_start="$data_start", every=2).logical_spec()
    root = next(iter(template.outputs.values()))
    store = FrameStore(tmp_path)
    runtimes = []
    for start in ["2024-01-02", "2024-01-03"]:
        runtime = ExecutionRuntime(materialization_store=store)
        bound = Graph.from_logical_spec(template, inputs={"feed.close": input_, "calendar": calendar},
                                       parameter_bindings={"$data_start": start})
        assert bound._single_output().logical_id == root
        assert bound.logical_spec().identity == template.identity
        # The first start is the coverage start; the second binding has an
        # explicitly restricted coverage Domain rather than a pre-anchor day.
        if start == "2024-01-02":
            bound.compute(runtime=runtime, dense_output=False)
        else:
            with pytest.raises(ValueError, match="absent from the anchored calendar"):
                bound.compute(runtime=runtime, dense_output=False)
        runtimes.append(runtime)
    assert runtimes[1].persistent_hits == 0


def test_rebalance_cache_retains_sparse_targets_and_all_decisions(tmp_path):
    input_ = source(traced=True)
    calendar = Panel.from_domain(input_.collect().with_columns(pl.lit(1.).alias("value")), input_.domain,
                                identity="calendar.v1", source_key="calendar")
    graph = rebalance(input_, calendar=calendar, data_start="2024-01-02", every=2)
    store = FrameStore(tmp_path)
    first_runtime = ExecutionRuntime(materialization_store=store)
    first = graph.compute(runtime=first_runtime, dense_output=False)
    root = graph._single_output().logical_id
    decisions = first_runtime.node_materializations[root].artifacts["decisions"]
    assert decisions["status"].to_list() == ["rebalance", "hold", "rebalance"]
    assert first.collect(dense=False).height == 4
    second_runtime = ExecutionRuntime(materialization_store=FrameStore(tmp_path, store.records))
    second = graph.compute(runtime=second_runtime, dense_output=False)
    assert second_runtime.persistent_hits == 1
    assert_frame_equal(first.collect(dense=False, include_traces=True), second.collect(dense=False, include_traces=True))
    assert second.collect(dense=False, include_traces=True)["available_date"].to_list() == [
        date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 4)]
    assert_frame_equal(second_runtime.node_artifacts[root]["decisions"], decisions)


def test_fit_audits_and_operator_checkpoints_are_replayed_on_hit(tmp_path):
    input_ = source()
    graph = rolling_elastic_net_prediction(input_, labels=input_, window=2,
        fit_every=1, min_samples=2, max_samples=20, label_maturity=1)
    store = FrameStore(tmp_path)
    with capture_training_audits() as first_audits, capture_operator_checkpoints(calendar=input_.domain.times) as first_states:
        graph.compute(runtime=ExecutionRuntime(materialization_store=store))
    with capture_training_audits() as second_audits, capture_operator_checkpoints(calendar=input_.domain.times) as second_states:
        runtime = ExecutionRuntime(materialization_store=FrameStore(tmp_path, store.records))
        graph.compute(runtime=runtime)
    assert first_audits and second_audits == first_audits
    assert second_states.captured == first_states.captured
    assert runtime.persistent_hits == 1


def test_budget_spills_existing_plans_without_recomputing_shared_operator(tmp_path):
    calls = []
    @transformer
    def counted(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(1)
        return frame
    upstream = counted(source())
    store = FrameStore(tmp_path)
    runtime = ExecutionRuntime(materialization_store=store)
    with resource_limits(ResourceLimits(cache_mib=0)):
        results = Graph(outputs=[rolling_mean(upstream, window=2, name="a"), negate(upstream, name="b")]).compute(
            runtime=runtime, dense_output=False)
    assert calls == [1]
    assert len(store.published) == 3
    assert runtime.materializations >= 3
    assert results["b"].collect()["value"].to_list() == [-1., -2., -3., -4., -5., -6.]


def test_sibling_spill_reuses_published_lazy_values_and_keeps_numerical_identity(tmp_path):
    """Count actual Polars execution when right recursion publishes the left."""
    from datetime import timedelta

    executed = []
    @transformer(contract=OperationContract(execution=ExecutionMode.LAZY))
    def lazy_counter(frame: pl.DataFrame) -> pl.DataFrame:
        def observe(batch):
            executed.append(batch.height)
            return batch
        return frame.map_batches(observe, streamable=False)

    domain = Domain(calendar=[date(2024, 1, 2)+timedelta(days=index) for index in range(10)],
        universe=[f"A{index:04d}" for index in range(1000)])
    frame = domain.grid_lazy().collect().with_columns(
        pl.Series("value", [float(index % 1000 + (index // 1000)*3) for index in range(domain.size)]))
    input_panel = Panel.from_domain(frame, domain, source_key="probe.source", identity="probe.source.v1")
    left = lazy_counter(input_panel, name="left")
    right = rolling_mean(negate(input_panel, name="right_base"), window=2, name="right")
    graph = add(left, right, name="root")
    logical = graph.logical_spec().to_dict()
    pressured_root, full_root = tmp_path/"pressure", tmp_path/"full"
    pressured_root.mkdir()
    full_root.mkdir()
    pressured = FrameStore(pressured_root)
    runtime = ExecutionRuntime(materialization_store=pressured)
    # Each plane is estimated at 400kB. Right's second operation spills the
    # previously captured left and right_base; the final add does not spill.
    with resource_limits(ResourceLimits(cache_mib=1)):
        sparse = graph.compute(runtime=runtime, dense_output=False)
    assert executed == [domain.size]
    actual = sparse.collect(dense=False)
    assert executed == [domain.size]
    assert runtime.materializations == 2
    assert runtime.persistent_misses == 4
    assert len(pressured.published) == 4
    expected = frame.with_columns((pl.col("value") + (-pl.col("value")).rolling_mean(window_size=2)
        .over("asset_id", order_by="time")).alias("value"))
    assert_frame_equal(actual, expected)
    full = FrameStore(full_root)
    with resource_limits(ResourceLimits(cache_mib=8)):
        unpressured = graph.compute(runtime=ExecutionRuntime(materialization_store=full), dense_output=False)
    assert_frame_equal(unpressured.collect(dense=False), expected)
    assert {key.node_id:key.identity for key in pressured.published} == {key.node_id:key.identity for key in full.published}
    assert graph.logical_spec().to_dict() == logical


def test_portable_identities_reject_invalid_public_boundary_values():
    from bagelquant_core import LogicalNodeSpec, MaterializationKey
    with pytest.raises(ValueError, match="source key"):
        LogicalNodeSpec.create(node_type="input", input_key="")
    with pytest.raises(ValueError, match="node type"):
        LogicalNodeSpec.create(node_type="unknown")
    with pytest.raises(ValueError, match="nonempty strings"):
        MaterializationKey.from_dict({"node_id": None, "implementation_id": "kernel",
                                     "inputs": [], "domain_identity": "domain"})
    with pytest.raises(ValueError):
        MaterializationLookup("unknown")
