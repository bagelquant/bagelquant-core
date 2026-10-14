"""Compact completion flags and interrupted frontiers use only metadata."""

from datetime import date
import sqlite3

import polars as pl
import pytest

from bagelquant_core import ExecutionRuntime, RevisionConflict
from bagelquant_core.operator import negate, rolling_mean
from test_global_graph import finish, global_graph, source


def test_repeat_uses_compact_flags_without_values_or_proofs(tmp_path, monkeypatch):
    value = source(70)
    graph, store, _ = global_graph(tmp_path, negate(rolling_mean(value, window=3)))
    original = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))

    def forbidden(*args, **kwargs):
        pytest.fail("metadata-only repeat consumed numerical inputs or ran a kernel")

    monkeypatch.setattr(store, "lookup", forbidden)
    monkeypatch.setattr(store, "read_values", forbidden)
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    monkeypatch.setattr(pl, "read_parquet", forbidden)
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(),
        input_loader=forbidden, input_proof=forbidden, input_window_loader=forbidden)
    assert plan.complete
    assert plan.ready("main") == ()
    assert plan.input_proof_windows("main").is_empty()
    assert finish(plan)["results"] == original["results"]
    assert all(usage["outcome"] == "up_to_date" for usage in plan.resource_usage["main"].values())
    with sqlite3.connect(store.meta_path) as connection:
        assert connection.execute("SELECT count(*) FROM node_completions").fetchone()[0] == 2
    assert len(store.node_completions("global", "main")) == 2


def test_interrupted_frontier_skips_successful_parent(tmp_path, monkeypatch):
    value = source(70)
    graph, store, root = global_graph(tmp_path, negate(rolling_mean(value, window=3)))
    interrupted = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    parent = interrupted.ready("main")[0]
    interrupted.execute("main", parent)
    assert store.graph_state("global")["current_update"] is None
    assert set(store.node_completions("global", "main")) == {parent}
    restarted = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
    assert restarted.ready("main") == (root,)
    assert restarted.results["main"] == interrupted.results["main"]
    original = ExecutionRuntime.run
    kernels = []

    def observed(runtime, local, *args, **kwargs):
        kernels.extend(node.logical_id for node in local.nodes if node.node_type == "operator")
        return original(runtime, local, *args, **kwargs)

    monkeypatch.setattr(ExecutionRuntime, "run", observed)
    finish(restarted)
    assert parent not in kernels
    assert set(kernels) == {root}
    assert set(store.node_completions("global", "main")) == {parent, root}


def test_same_day_revision_and_context_do_not_use_stale_flags(tmp_path):
    value = source(70)
    graph, store, _ = global_graph(tmp_path, negate(value))
    finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    revision = source(70, revision=40)
    changed = graph.plan_update({"main": {"x": revision}}, through=revision.domain.times.max())
    assert not changed.complete
    assert changed.results["main"] == {}
    graph.register_context("other", anchor="2024-01-01")
    separate = graph.plan_update({"other": {"x": value}}, through=value.domain.times.max(), roots=list(graph._state()["spec"].outputs))
    assert not separate.complete
    assert store.node_completions("global", "other") == {}
    finish(separate)
    assert len(store.node_completions("global", "other")) == 1


def test_historical_branch_preserves_latest_flag_and_publication_fence(tmp_path):
    initial = source(3)
    graph, store, root = global_graph(tmp_path, negate(initial))
    finish(graph.plan_update({"main": {"x": initial}}, through=initial.domain.times.max()))
    latest = source(4)
    current = finish(graph.plan_update({"main": {"x": latest}}, through=latest.domain.times.max()))
    historical = graph.plan_update({"main": {"x": initial}}, through=initial.domain.times.max(), roots=[root])
    finish(historical)
    row = store.node_completions("global", "main")[root]
    assert row["through"] == str(date(2024, 1, 4))
    assert row["identity"] == current["results"]["main"][root]
    plan = graph.plan_update({"main": {"x": latest}}, through=latest.domain.times.max())
    graph.sleep(root)
    with pytest.raises(RevisionConflict):
        plan.publish()
