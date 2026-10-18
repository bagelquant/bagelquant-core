"""Count-only preparation must leave admission, work and authorities unchanged."""
from datetime import date, timedelta
from dataclasses import replace
from types import SimpleNamespace

import polars as pl
import pytest

from bagelquant_core import CoreStore, Domain, MaterializationKey, Node
import bagelquant_core._finite_cold_batch as cold
import bagelquant_core.incremental as incremental
from bagelquant_core.operator import negate
from bagelquant_core.resources import active_resource_limits, resource_limits
from test_finite_cold_batches import (
    LIMITS, NATIVE, assert_preserved, chain, execute, file_state, scientific_inventory,
)
from test_finite_cold_planning import controller
from test_finite_pct_view import bits, indexed_source_proof, source
from test_global_graph import finish


def predicate_rows(domain, start, end):
    """The untouched parent's inclusive active-grid UInt64 reduction."""
    return int(domain.grid_lazy().select(pl.col("time").is_between(start, end)
        .cast(pl.UInt64).sum()).collect().item())


def filtered_sessions(calendar, start, end):
    axis = pl.Series("time", calendar, dtype=pl.Date)
    return len(axis.filter(axis.is_between(start, end)))


def gapped_source(count):
    original = source(count)
    old_days = original.domain.times.to_list()
    days = [old_days[0] + timedelta(days=2 * i) for i in range(count)]
    mapping = dict(zip(old_days, days, strict=True))
    members = original.domain.grid_lazy().collect()
    members = members.with_columns(pl.Series("time", [mapping[day] for day in members["time"]], dtype=pl.Date))
    frame = original.collect(dense=False, include_traces=True)
    frame = frame.with_columns(pl.Series("time", [mapping[day] for day in frame["time"]], dtype=pl.Date))
    extra_day = days[0] + timedelta(days=1)
    members = pl.concat([members, pl.DataFrame({"time": [extra_day], "asset_id": ["gap-only-资产"]})])
    extra = pl.DataFrame([(extra_day, "gap-only-资产", 5.0, extra_day, extra_day)],
        schema=frame.schema, orient="row")
    domain = Domain(calendar=days, universe=members.with_columns(pl.lit(True).alias("active")))
    return Node.from_domain(pl.concat([frame, extra]), domain, source_key="x", trace_columns=original.trace_columns)


@pytest.mark.parametrize("batch_rows", [100, 512, 100_000])
@pytest.mark.parametrize("memory", [17, 512])
def test_candidate_bounds_raw_reserve_and_limits_match_parent_predicates(monkeypatch, batch_rows, memory):
    days = [date(2020, 1, 2) + timedelta(days=2 * i) for i in range(129)]
    keys = [(day, "A") for day in days]
    keys += [(days[0] + timedelta(days=1), "off-calendar-资产"),
        (days[90], "retired-and-returning"), (days[-1], "retired-and-returning")]
    domain = Domain(calendar=days, universe=pl.DataFrame(keys, orient="row",
        schema={"time": pl.Date, "asset_id": pl.String}).with_columns(pl.lit(True).alias("active")))
    parts = [{"start": None, "end": None, "rows": 700},
        {"start": str(days[96]), "end": str(days[-1]), "rows": 900}]
    limits = replace(LIMITS, batch_rows=batch_rows, memory_target_mib=memory,
        cache_mib=min(LIMITS.cache_mib, memory - 2))
    initial_limits = active_resource_limits()
    def run():
        calls = []
        def start(origin, end):
            calls.append((origin, end))
            return days[max(0, origin - 9)] - timedelta(days=1)
        with controller(domain, limits=limits, parts=parts,
                traces=("available_date", "observed_at")) as value:
            selected = [value.plan(begin, days, start, 32) for begin in (0, 32, 96)]
            return selected, calls, dict(value._read_starts), value.planning_bytes
    with resource_limits(initial_limits):
        actual = run()
        with monkeypatch.context() as patch:
            patch.setattr(cold._ColdFiniteBatches, "_output_rows",
                lambda value, start, end: predicate_rows(value._domain, start, end))
            patch.setattr(cold, "_inclusive_date_count", filtered_sessions)
            expected = run()
        assert actual == expected
        assert active_resource_limits() is initial_limits


def test_unequal_parent_calendar_is_rejected_before_manifest_authority(tmp_path, monkeypatch):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(3)]
    def domain(calendar):
        return Domain(calendar=calendar, universe=pl.DataFrame({"time": calendar,
            "asset_id": ["A"] * len(calendar), "active": [True] * len(calendar)}))
    output, parent = domain(days), domain(days[:2])
    key = MaterializationKey("parent", "operator.v1", ("source",), parent.signature)
    value = Node.from_domain(parent.empty_frame(), parent, identity=key.identity)
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "artifacts")
    monkeypatch.setattr(store, "manifest", lambda *_args: pytest.fail("unequal calendar reached saved authority"))
    node = SimpleNamespace(node_id="new", operator=negate.registry_name,
        inputs=("parent",), parameters={}, panel_parameters={})
    with cold._ColdFiniteBatches(node, {"parent": value}, output,
            {"history_observations": 0}, store, LIMITS, {"parent": key}) as value:
        assert not value.eligible


@pytest.mark.parametrize("error", [MemoryError("count"), KeyboardInterrupt()])
def test_scalar_failure_propagates_before_history_or_allocation(monkeypatch, error):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(96)]
    domain = Domain(calendar=days, universe=pl.DataFrame({"time": days,
        "asset_id": ["A"] * len(days), "active": [True] * len(days)}))
    calls = []
    def broken(*_args):
        calls.append("count")
        raise error
    monkeypatch.setattr(Domain, "_count_rows", broken)
    value = controller(domain)
    with pytest.raises(type(error)) as captured:
        with value:
            value.plan(0, days, lambda *_args: pytest.fail("failed count queried history"), 32)
    assert captured.value is error and calls == ["count"]
    assert value._domain is value._parents is None
    assert not value._read_starts and not value._parent_bounds


@NATIVE
@pytest.mark.parametrize("kind,change,gapped", [
    ("pct", "cold", False), ("smooth", "cold", False), ("rank", "cold", False),
    ("pct", "append", False), ("smooth", "append", False), ("rank", "append", False),
    ("pct", "repair", False), ("smooth", "repair", False), ("rank", "repair", False),
    ("pct", "cold", True), ("smooth", "cold", True), ("rank", "cold", True),
])
def test_public_work_windows_typed_bits_and_history_match_parent_count_path(tmp_path, monkeypatch, kind, change, gapped):
    """Only the count oracles change; all public Graph/operator paths stay live."""
    window, read = Domain.window, CoreStore.read_values
    def run(root):
        windows, reads = [], []
        def selected(owner, first, last):
            result = window(owner, first, last)
            windows.append((owner.signature, first, last, result.signature, result.size))
            return result
        def observed(store, identity, **kwargs):
            reads.append((identity, kwargs["start"], kwargs["end"], active_resource_limits()))
            return read(store, identity, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(Domain, "window", selected)
            patch.setattr(CoreStore, "read_values", observed)
            x = (gapped_source if gapped else source)(257 if change == "cold" else 96)
            graph, store, output, plan, result = execute(root, x, expression=chain(x, kind))
            before = file_state(store)
            if change != "cold":
                x = source(160) if change == "append" else source(96, revision=70)
                graph.merge(chain(x, kind))
                plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(),
                    input_proof=indexed_source_proof(x))
                result = finish(plan, limits=LIMITS)
                assert_preserved(store, before)
                assert plan.resource_usage["main"][output]["reused_blocks"] > 0
            inventory = scientific_inventory(store)
            frame = bits(store.read(result["results"]["main"][output]).panel.collect(
                dense=False, include_traces=True))
            before_repeat = file_state(store)
            prior_reads = list(reads)
            repeat = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(),
                input_proof=indexed_source_proof(x))
            assert repeat.complete
            assert finish(repeat, limits=LIMITS)["results"] == result["results"]
            assert file_state(store) == before_repeat
            assert reads == prior_reads
            return (result["results"], plan.resource_usage, inventory, frame, windows, reads)
    actual = run(tmp_path / "scalar")
    with monkeypatch.context() as patch:
        patch.setattr(cold._ColdFiniteBatches, "_output_rows",
            lambda value, start, end: predicate_rows(value._domain, start, end))
        patch.setattr(cold, "_inclusive_date_count", filtered_sessions)
        def old_counter(calendar, start, end):
            # Independent public calendar-window oracle for the three counts.
            # Call the original window directly so the real-window spy above
            # continues to compare only required authority/Node scopes.
            return len(window(Domain(calendar=calendar, universe=["counter-oracle"]), start, end).times)
        patch.setattr(incremental, "_calendar_sessions", old_counter)
        expected = run(tmp_path / "parent-counts")
    assert actual == expected
