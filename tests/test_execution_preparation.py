"""Preparation sharing preserves canonical arithmetic and publication fences."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import sqlite3
from threading import Barrier

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import CoreStore, Domain, Node, RevisionConflict
from bagelquant_core.operation_contract import ExecutionMode, OperationContract, TraceRule
from bagelquant_core.incremental import _canonical_halos
from bagelquant_core.operator import add, negate, operator, rolling_mean, rolling_std
from bagelquant_core.preparation import ExecutionPreparation, preparation_scope
from bagelquant_core.resources import ResourceLimits, resource_limits
from test_global_graph import finish, global_graph


def retired_source(count=128):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]
    members = [(day, "A") for day in days] + [(day, "RETIRED") for day in days[:2]]
    keys = pl.DataFrame(members, schema={"time": pl.Date, "asset_id": pl.String}, orient="row")
    domain = Domain(calendar=days, universe=keys.with_columns(pl.lit(True).alias("active")))
    frame = keys.with_columns(
        ((pl.col("time").cast(pl.Int32) % 37).cast(pl.Float64) / 3 + 2).alias("value"),
        (pl.col("time") + timedelta(days=1)).alias("available_date"))
    return Node.from_domain(frame, domain, source_key="x", trace_columns=("available_date",))


def test_shared_halos_and_slabs_reduce_reads_with_identical_canonical_kernels(tmp_path):
    value = retired_source()
    intermediate = negate(value)
    expression = add(rolling_mean(intermediate, window=10), rolling_std(intermediate, window=10))
    reports, outputs, identities = [], [], []
    for name, cache in (("bulk", 0), ("shared", 16)):
        limits = ResourceLimits(cache_mib=cache)
        with resource_limits(limits):
            graph, store, root = global_graph(tmp_path / name, expression)
            plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
            # Scope planning and kernels share one schedule, without value work.
            plan.input_proof_windows("main")
            receipt = finish(plan, limits=limits)
        identities.append(receipt["results"]["main"][root])
        outputs.append(store.read_values(identities[-1], include_traces=True))
        reports.append((plan.preparation_usage, plan.resource_usage["main"]))
    assert identities[0] == identities[1]
    assert_frame_equal(outputs[0], outputs[1], check_exact=True)
    assert reports[0][1] == reports[1][1]
    assert reports[1][0]["halo_membership_sorts"] == 1
    assert reports[1][0]["parent_decoded_sessions"] == 512
    assert reports[0][0]["parent_decoded_sessions"] == 1024
    assert reports[1][0]["domain_window_hits"] > 0
    assert reports[1][0]["parent_slab_hits"] > 0
    assert reports[1][0]["resident_bytes"] == 0  # publication expires numerical preparation


def test_halo_schedule_preserves_each_history_and_retires_with_operation():
    domain = retired_source().domain
    histories = (0, 1, 9, 719)
    original = {history: _canonical_halos(domain, history) for history in histories}
    preparation = ExecutionPreparation(1024 * 1024)
    for history in histories:
        preparation.register_halo(domain, history)
    with preparation_scope(preparation):
        assert {history: _canonical_halos(domain, history) for history in histories} == original
        assert preparation.report()["halo_membership_sorts"] == 1
    preparation.clear()
    with preparation_scope(preparation):
        assert _canonical_halos(domain, 9) == original[9]
    assert preparation.report()["halo_membership_sorts"] == 2


def test_actual_domains_and_bounded_eviction_cannot_hide_opaque_identity():
    days = [date(2024, 1, 1), date(2024, 1, 2)]
    first = Domain(calendar=days, universe=["A"], identity="opaque")
    other = Domain(calendar=days, universe=["B"], identity="opaque")
    preparation = ExecutionPreparation(4096)
    with preparation_scope(preparation):
        selected = first.window(*days)
        assert first.window(*days) is selected
        assert not other.window(*days).asset_ids.equals(selected.asset_ids)
        for index in range(20):
            preparation.remember("fixture", index, lambda: "x" * 1000, len)
        assert preparation.report()["evictions"] > 0
        assert preparation.bytes <= preparation.maximum_bytes
        preparation.restrict(0)
        assert preparation.bytes == 0
    assert first.window(*days) is not selected


def test_absent_proofs_are_not_cached():
    preparation = ExecutionPreparation(4096)
    available = [None]
    assert preparation.remember("proof", "x", lambda: available[0], len) is None
    available[0] = "published"
    assert preparation.remember("proof", "x", lambda: available[0], len) == "published"


def test_unrelated_preparation_misses_can_overlap():
    preparation = ExecutionPreparation(4096)
    barrier = Barrier(2)
    def prepare(key):
        def build():
            barrier.wait(timeout=3)
            return key
        return preparation.remember("parallel", key, build, len)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(prepare, key) for key in ("a", "b")]
        assert [job.result(timeout=5) for job in jobs] == ["a", "b"]


def test_publication_context_amortizes_initialization_and_checks_schema(tmp_path, monkeypatch):
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "values")
    initialize = store._initialize
    calls = []
    def counted():
        calls.append(1)
        initialize()
    monkeypatch.setattr(store, "_initialize", counted)
    with store.publication_context():
        store.initialize()
        with store.publication_context():
            store.initialize()
        assert len(calls) == 1
        with sqlite3.connect(store.meta_path) as db:
            db.execute("PRAGMA user_version=2")
        with pytest.raises(ValueError, match="incompatible"):
            with store.transaction():
                pass
    with pytest.raises(ValueError, match="incompatible"):
        store.initialize()
    assert len(calls) == 2


def test_publication_context_rejects_replaced_metadata_file(tmp_path):
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "values")
    replacement = CoreStore(tmp_path / "replacement.sqlite", tmp_path / "replacement-values")
    replacement.initialize()
    with store.publication_context():
        replacement.meta_path.replace(store.meta_path)
        with pytest.raises(RevisionConflict, match="changed"):
            store.initialize()
        with pytest.raises(RevisionConflict, match="changed"):
            with store.transaction():
                pass


def test_native_admission_covers_unit_and_never_enters_for_noop(tmp_path):
    phases, active = [], []
    @contextmanager
    def admit(kind):
        phases.append(kind)
        active.append(kind)
        try:
            yield
        finally:
            active.pop()
    @operator(contract=OperationContract(execution=ExecutionMode.EAGER_BARRIER,
        trace_rule=TraceRule.PASSTHROUGH))
    def custom(frame: pl.DataFrame) -> pl.DataFrame:
        assert active == ["mixed"]
        return frame
    value = retired_source(3)
    graph, _store, _root = global_graph(tmp_path, negate(custom(value)))
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), native_admission=admit)
    finish(plan)
    assert phases == ["mixed", "columnar"]
    repeated = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), native_admission=admit)
    assert repeated.complete
    finish(repeated)
    assert phases == ["mixed", "columnar"]
