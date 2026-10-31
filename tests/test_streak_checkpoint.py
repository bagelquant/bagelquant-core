"""Streak state preserves the original formula and skips completed transitions."""
from datetime import date, timedelta
from dataclasses import replace

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from bagelquant_core import Domain, ExecutionRuntime, Node
from bagelquant_core.operator import streak_count
from bagelquant_core.operator.streaks import _streak_state, _run_trace
from bagelquant_core.streak_state import valid_streak_state
from test_global_graph import global_graph, finish


def source(count, *, dynamic=False, traces=True, revision=None, empty_tail=False):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]
    sequence = [1., 2., 3., 2., 1., 1., 0., None, -1., float("nan"), 2., 2., 4., 3., 3., 2.]
    rows = []
    for i, day in enumerate(days):
        for j, asset in enumerate(("a", "b", "c")):
            if dynamic and (asset == "b" and i % 14 in range(3, 10)
                    or asset == "c" and i > 22 and i < 34
                    or empty_tail and i == count - 1):
                continue
            value = sequence[(i + j * 2) % len(sequence)]
            if revision is not None and i == revision:
                value = 6.
            rows.append((day, asset, value, days[min(i, i % 7)], days[min(i, (i + 2) % 5)]))
    frame = pl.DataFrame(rows, schema={"time": pl.Date, "asset_id": pl.String,
        "value": pl.Float64, "available_date": pl.Date, "observation_date": pl.Date}, orient="row")
    domain = Domain(calendar=days, universe=(frame.select("time", "asset_id").with_columns(
        pl.lit(True).alias("active")) if dynamic else ["a", "b", "c"]))
    return Node.from_domain(frame if traces else frame.select("time", "asset_id", "value"), domain,
        source_key="x", trace_columns=("available_date", "observation_date") if traces else ())


def expected(value, reset_on_equal):
    frame = value.collect(dense=True, include_traces=True).sort("asset_id", "time")
    state = _streak_state(frame, reset_on_equal=reset_on_equal)
    result = state.select("time", "asset_id", "value")
    if value.trace_columns:
        result = result.join(_run_trace(state, traces=value.trace_columns),
            on=["time", "asset_id"], how="left")
    return result.sort("time", "asset_id")


def execute(graph, value, *, reads=None):
    def window(_context, _name, first, last):
        if reads is not None:
            reads.append((first, last))
        return Node.from_domain(value.lazy(include_traces=True).filter(pl.col("time").is_between(first, last)),
            value.domain.window(first, last), identity=value.identity,
            trace_columns=value.trace_columns, trace_identity=value.trace_identity)
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(),
        input_window_loader=window, input_loader=lambda *_: pytest.fail("streak read complete historical inputs"))
    return plan, finish(plan)


@pytest.mark.parametrize("reset_on_equal", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("traces", [False, True])
def test_append_matches_original_values_and_traces_without_historical_work(tmp_path, monkeypatch, reset_on_equal, dynamic, traces):
    first = source(32, dynamic=dynamic, traces=traces)
    graph, store, root = global_graph(tmp_path, streak_count(first, reset_on_equal=reset_on_equal))
    initial = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    old = store.manifest(initial["results"]["main"][root])
    originals = {part["$file"]: (store.artifact_path / part["$file"]).read_bytes() for part in old["partitions"]}
    import bagelquant_core.operator.streaks as operators
    calls = []
    ordinary = operators._streak_state
    def observed(frame, **kwargs):
        calls.append((frame["time"].min(), frame["time"].max()))
        return ordinary(frame, **kwargs)
    monkeypatch.setattr(operators, "_streak_state", observed)
    for count in (33, 36):
        current = source(count, dynamic=dynamic, traces=traces)
        reads = []
        before = first.domain.times.max()
        plan, receipt = execute(graph, current, reads=reads)
        usage = plan.resource_usage["main"][root]
        assert usage["checkpoint_through"] == str(before)
        assert usage["replayed_sessions"] == count - len(first.domain.times)
        assert reads and all(first_day > before for first_day, _ in reads)
        assert calls and all(first_day > before for first_day, _ in calls)
        actual = store.read(receipt["results"]["main"][root]).panel.collect(dense=True, include_traces=True)
        assert_frame_equal(actual, expected(current, reset_on_equal), check_exact=True)
        assert valid_streak_state(store.read_checkpoint(receipt["results"]["main"][root])["state"])
        for name, content in originals.items():
            assert (store.artifact_path / name).read_bytes() == content
        first = current
        calls.clear()
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("same-boundary repeat ran a kernel"))
    plan, repeated = execute(graph, first, reads=[])
    assert repeated["results"] == receipt["results"]
    assert plan.resource_usage["main"][root]["outcome"] == "up_to_date"


@pytest.mark.parametrize("reset_on_equal", [False, True])
def test_empty_session_retains_departed_asset_state_for_reentry(tmp_path, reset_on_equal):
    first = source(32, dynamic=True)
    graph, store, root = global_graph(tmp_path, streak_count(first, reset_on_equal=reset_on_equal))
    initial = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    current = source(33, dynamic=True, empty_tail=True)
    _, empty = execute(graph, current)
    checkpoint = store.read_checkpoint(empty["results"]["main"][root])["state"]
    assert checkpoint["through"] == str(current.domain.times.max())
    assert checkpoint["assets"] == store.read_checkpoint(initial["results"]["main"][root])["state"]["assets"]
    # Keep the empty session in the original membership while introducing reentry.
    later = source(36, dynamic=True)
    frame = later.collect(dense=True, include_traces=True).filter(pl.col("time") != current.domain.times.max())
    domain = Domain(calendar=later.domain.times, universe=frame.select("time", "asset_id").with_columns(pl.lit(True).alias("active")))
    later = Node.from_domain(frame, domain, source_key="x", trace_columns=later.trace_columns)
    _, final = execute(graph, later)
    assert_frame_equal(store.read(final["results"]["main"][root]).panel.collect(dense=True, include_traces=True),
        expected(later, reset_on_equal), check_exact=True)


@pytest.mark.parametrize("reset_on_equal", [False, True])
def test_revision_uses_earlier_proven_checkpoint_and_unknown_history_blocks(tmp_path, monkeypatch, reset_on_equal):
    first = source(32)
    graph, store, root = global_graph(tmp_path, streak_count(first, reset_on_equal=reset_on_equal))
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    execute(graph, source(36))
    revised = source(36, revision=33)
    reads = []
    plan, repaired = execute(graph, revised, reads=reads)
    assert plan.resource_usage["main"][root]["replayed_sessions"] == 4
    assert all(first_day > first.domain.times.max() for first_day, _ in reads)
    assert_frame_equal(store.read(repaired["results"]["main"][root]).panel.collect(dense=True, include_traces=True),
        expected(revised, reset_on_equal), check_exact=True)
    old_pointer = store.graph_state("global")["current_update"]
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("unproven history was replayed"))
    with pytest.raises(ValueError, match="proven preceding state"):
        execute(graph, source(37, revision=0))
    assert store.graph_state("global")["current_update"] == old_pointer


def test_bad_checkpoint_is_rejected_before_historical_kernel(tmp_path, monkeypatch):
    first = source(32)
    graph, store, _ = global_graph(tmp_path, streak_count(first))
    initial = finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    read = store.read
    def damaged(identity):
        item = read(identity)
        return replace(item, checkpoint={**item.checkpoint, "state": {**item.checkpoint["state"], "assets": {"a": {}}}})
    monkeypatch.setattr(store, "read", damaged)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("damaged state replayed history"))
    with pytest.raises(ValueError, match="proven preceding state"):
        execute(graph, source(33))
    assert store.graph_state("global")["current_update"] == initial["identity"]


@pytest.mark.parametrize("off_calendar", [date(2023, 12, 31), date(2024, 1, 2), date(2024, 1, 4)])
def test_off_calendar_streak_state_blocks_during_planning_before_any_kernel(tmp_path, monkeypatch, off_calendar):
    from bagelquant_core.operation_contract import ExecutionMode
    from bagelquant_core.operator import negate
    days = [date(2024, 1, 1), date(2024, 1, 3)]
    keys = sorted({*days, off_calendar})
    frame = pl.DataFrame({"time": keys, "asset_id": ["a"] * len(keys),
        "value": [float(index) for index in range(len(keys))]})
    domain = Domain(calendar=days, universe=frame.select("time", "asset_id").with_columns(pl.lit(True).alias("active")))
    first = Node.from_domain(frame, domain, source_key="x")
    graph, store, _ = global_graph(tmp_path, streak_count(negate(first)))
    with monkeypatch.context() as legacy:
        legacy.setattr(streak_count, "contract", replace(streak_count.contract, execution=ExecutionMode.LAZY,
            checkpoint_replay=False, checkpoint_history=None, checkpoint_validator=None))
        old = finish(graph.plan_update({"main": {"x": first}}, through=days[-1]))
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("off-calendar planning ran a kernel"))
    with pytest.raises(ValueError, match="outside the execution calendar"):
        graph.plan_update({"main": {"x": first}}, through=days[-1],
            input_loader=lambda *_: pytest.fail("off-calendar planning consumed historical inputs"))
    assert store.graph_state("global")["current_update"] == old["identity"]
