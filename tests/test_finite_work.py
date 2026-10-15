"""Fresh-process finite math preserves keys, prefixes and coordinator authority."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import FiniteKernelResult, FiniteKernelWork, compute_finite_work
from bagelquant_core.operator import negate, rolling_mean
from test_execution_preparation import retired_source
from test_global_graph import finish, global_graph


def test_spawned_canonical_work_matches_inline_and_reuses_append_prefix(tmp_path):
    before, current = retired_source(66), retired_source(70)
    reference, reference_store, reference_root = global_graph(tmp_path / "reference", negate(rolling_mean(current, window=10)))
    completed = finish(reference.plan_update({"main": {"x": current}}, through=current.domain.times.max()))
    expected = completed["results"]["main"][reference_root]
    graph, store, root = global_graph(tmp_path / "spawn", negate(rolling_mean(before, window=10)))
    stages = []
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        def dispatch(work):
            # Exercise the real serialized API, not a shared in-memory object.
            restored = FiniteKernelWork.from_dict(work.to_dict())
            assert restored == work
            stages.append(work.staging_root)
            result = pool.submit(compute_finite_work, restored).result(timeout=30)
            return FiniteKernelResult.from_dict(result.to_dict())
        first = graph.plan_update({"main": {"x": before}}, through=before.domain.times.max(), kernel_dispatch=dispatch)
        old = finish(first)["results"]["main"][root]
        old_manifest = store.manifest(old)
        appended = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(), kernel_dispatch=dispatch)
        actual = finish(appended)["results"]["main"][root]
        assert actual == expected
        assert store.manifest(old) == old_manifest
        assert_frame_equal(store.read_values(actual, include_traces=True),
            reference_store.read_values(expected, include_traces=True), check_exact=True)
        assert all(usage["computed_output_sessions"] == 4 for usage in appended.resource_usage["main"].values())
        assert appended.preparation_usage["dispatched_kernels"] == 2
        dispatched = len(stages)
        repeated = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(), kernel_dispatch=dispatch)
        assert repeated.complete
        finish(repeated)
        assert len(stages) == dispatched
    assert all(not Path(directory).exists() for directory in stages)


def test_invalid_worker_result_cannot_publish_or_leave_staging(tmp_path):
    value = retired_source(3)
    graph, store, root = global_graph(tmp_path, negate(value))
    stages = []
    def invalid(work):
        stages.append(work.staging_root)
        return replace(compute_finite_work(work), rows=1000)
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), kernel_dispatch=invalid)
    with pytest.raises(ValueError, match="row count"):
        plan.execute("main", root)
    assert not plan.complete
    assert store.node_completions("global", "main") == {}
    assert store.graph_state("global")["current_update"] is None
    assert all(not Path(directory).exists() for directory in stages)
    retried = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), kernel_dispatch=compute_finite_work)
    finish(retried)
    assert retried.complete


def test_cancel_after_worker_compute_does_not_publish(tmp_path):
    value = retired_source(3)
    graph, store, root = global_graph(tmp_path, negate(value))
    canceled, stages = [], []
    def dispatch(work):
        stages.append(work.staging_root)
        result = compute_finite_work(work)
        canceled.append(True)
        return result
    def check():
        if canceled:
            raise InterruptedError("canceled")
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), kernel_dispatch=dispatch)
    with pytest.raises(InterruptedError):
        plan.execute("main", root, check_canceled=check)
    assert store.node_completions("global", "main") == {}
    assert all(not Path(directory).exists() for directory in stages)
