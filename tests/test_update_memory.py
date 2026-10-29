"""Public admission sizing consumes metadata and preserves finite numerics."""
from dataclasses import replace
from contextlib import contextmanager

import polars as pl
import pytest

from bagelquant_core.operator import negate, rolling_mean
from bagelquant_core.resources import ResourceLimits
from test_global_graph import finish, global_graph, source


def test_memory_preflight_is_metadata_only_and_covers_execution(tmp_path, monkeypatch):
    value = source(70, dynamic=True)
    graph, store, root = global_graph(tmp_path, rolling_mean(value, window=40))
    phases = []
    @contextmanager
    def native(kind):
        phases.append((kind, "enter"))
        try:
            yield
        finally:
            phases.append((kind, "exit"))
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), native_admission=native)
    limits = ResourceLimits(memory_target_mib=16, cache_mib=0)
    def forbidden(*args, **kwargs):
        pytest.fail("memory preflight consumed values or published results")
    with monkeypatch.context() as preflight:
        for name in ("read_values", "lookup", "publish"):
            preflight.setattr(store, name, forbidden)
        preflight.setattr(plan, "_load_saved_parents", forbidden)
        required = plan.required_memory_bytes("main", root, limits=limits)
    assert isinstance(required, int) and required > 0
    assert phases == [("columnar", "enter"), ("columnar", "exit")]
    assert not store.inventory()
    finish(plan, limits=replace(limits, memory_target_mib=(required + 1024**2 - 1) // 1024**2))
    previous_phases = list(phases)
    assert plan.required_memory_bytes("main", root, limits=limits) == 0
    assert phases == previous_phases


@pytest.mark.parametrize("factory", [lambda value: negate(negate(value)), lambda value: rolling_mean(value, window=4)])
def test_retained_preflight_sizes_only_missing_append(tmp_path, monkeypatch, factory):
    before, current = source(66), source(67)
    graph, store, root = global_graph(tmp_path, factory(before))
    old = finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max(),
        input_proof=lambda _context, _key, first, last: "same"))
    original = store.manifest(old["results"]["main"][root])
    limits = ResourceLimits(cache_mib=0)
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
        input_proof=lambda _context, _key, first, last: "same")
    # A second independent cold graph gives a resource comparison, not a value oracle.
    cold, reference, cold_root = global_graph(tmp_path / "cold", factory(current))
    initial = cold.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
        input_proof=lambda _context, _key, first, last: "same")
    assert plan.required_memory_bytes("main", root, limits=limits) < initial.required_memory_bytes("main", cold_root, limits=limits)
    assert store.manifest(old["results"]["main"][root]) == original
    from polars.testing import assert_frame_equal
    saved = finish(plan)
    expected = finish(initial)
    assert saved["results"]["main"][root] == expected["results"]["main"][cold_root]
    assert_frame_equal(store.read_values(saved["results"]["main"][root], include_traces=True),
        reference.read_values(expected["results"]["main"][cold_root], include_traces=True), check_exact=True)
    assert plan.resource_usage["main"][root]["computed_output_sessions"] == 1
    assert store.manifest(old["results"]["main"][root]) == original


def test_missing_external_metadata_proof_returns_unknown_without_payload_fallback(tmp_path, monkeypatch):
    before, current = source(66), source(67)
    graph, _, root = global_graph(tmp_path, rolling_mean(before, window=4))
    finish(graph.plan_update({"main": {"x": before}}, through=before.domain.times.max()))
    plan = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max())
    def forbidden(*args, **kwargs):
        pytest.fail("preflight hashed numerical source values")
    monkeypatch.setattr(current, "interval_identity", forbidden)
    assert plan.required_memory_bytes("main", root, limits=ResourceLimits()) is None


def test_invalid_or_unknown_node_does_not_claim_finite_memory(tmp_path):
    value = source(3)
    graph, _, root = global_graph(tmp_path, negate(value))
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    with pytest.raises(KeyError):
        plan.required_memory_bytes("main", "missing", limits=ResourceLimits())


def test_near_budget_static_kernel_is_not_charged_for_ineligible_preparation(tmp_path):
    from datetime import date, timedelta
    from bagelquant_core import Domain, Node
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(32)]
    domain = Domain(calendar=dates, universe=[f"A{i}" for i in range(170)])
    value = Node.from_domain(domain.grid_lazy().with_columns(pl.lit(1.).alias("value")),
        domain, source_key="x", identity="static")
    graph, _, root = global_graph(tmp_path, negate(value))
    plan = graph.plan_update({"main": {"x": value}}, through=dates[-1])
    limits = ResourceLimits(memory_target_mib=4, cache_mib=0)
    assert 3 * 1024**2 < plan.required_memory_bytes("main", root, limits=limits) <= 4 * 1024**2
    finish(plan, limits=limits)
