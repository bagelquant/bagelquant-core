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
from bagelquant_core.preparation import ExecutionPreparation, PreparationMetadataMemo, preparation_scope
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


def test_diagnostics_distinguish_evictions_and_uncached_oversize_builds():
    preparation = ExecutionPreparation(2048)
    for key in ("first", "second", "first"):
        preparation.remember("parent_slab", key, lambda: b"x" * 512, len)
    preparation.remember("domain_window", "large", lambda: b"x" * 2048, len)
    report = preparation.report()
    assert report["parent_slab_builds"] == 3
    assert report["parent_slab_evictions"] == report["evictions"] == 2
    assert report["parent_slab_evicted_bytes"] == 3072
    assert report["domain_window_oversize"] == 1
    assert report["domain_window_oversize_bytes"] == 3072
    assert report["resident_bytes"] == 1536


def test_interval_metadata_uses_one_existing_reserve_and_fresh_public_trees(tmp_path):
    value = retired_source(65)
    graph, store, root = global_graph(tmp_path, negate(value))
    receipt = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    identity = receipt["results"]["main"][root]
    expected = store.interval_identity(identity, start=value.domain.times.min(), end=value.domain.times.max())
    counts = []
    for budget in (0, 1024 * 1024):
        preparation = ExecutionPreparation(budget)
        events = []
        store._metadata_observer = lambda name, amount: events.extend([name] * amount)
        try:
            with preparation_scope(preparation):
                store.manifest(identity)["key"]["node_id"] = "caller mutation"
                assert store.manifest(identity)["node_id"] == root
                for _ in range(2):
                    assert store.interval_identity(identity, start=value.domain.times.min(),
                        end=value.domain.times.max()) == expected
                assert preparation.bytes <= budget
                counts.append((events.count("manifest_reads"), events.count("interval_index_reads")))
            preparation.clear()
            assert preparation.bytes == 0
        finally:
            store._metadata_observer = None
    assert counts == [(4, 2), (1, 1)]


def test_preparation_metadata_expiry_missing_proof_and_explicit_audit(tmp_path):
    import json
    value = retired_source(32)
    graph, store, root = global_graph(tmp_path, negate(value))
    receipt = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    identity = receipt["results"]["main"][root]
    preparation = ExecutionPreparation(1024 * 1024)
    with preparation_scope(preparation):
        assert store.manifest("missing") is None
        assert not preparation._values
        store.describe(identity)
        with sqlite3.connect(store.meta_path) as db:
            manifest = json.loads(db.execute("SELECT manifest_json FROM materializations WHERE identity=?",
                (identity,)).fetchone()[0])
            manifest["through"] = "2099-01-01"
            db.execute("UPDATE materializations SET manifest_json=? WHERE identity=?",
                (json.dumps(manifest), identity))
        with pytest.raises(ValueError):
            store.verify_identity(identity)
        preparation.clear()
        with pytest.raises(ValueError, match="corrupt Core manifest"):
            store.manifest(identity)


def test_preparation_metadata_namespaces_stores_and_never_caches_absent_index(tmp_path):
    value = retired_source(32)
    stores = []
    for name in ("first", "second"):
        graph, store, root = global_graph(tmp_path / name, negate(value))
        receipt = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
        stores.append((store, receipt["results"]["main"][root]))
    with sqlite3.connect(stores[1][0].meta_path) as db:
        saved = db.execute("SELECT * FROM computation_index WHERE identity=?", (stores[1][1],)).fetchone()
        db.execute("DELETE FROM computation_index WHERE identity=?", (stores[1][1],))
    preparation = ExecutionPreparation(1024 * 1024)
    with preparation_scope(preparation):
        assert stores[0][0].interval_identity(stores[0][1], start=value.domain.times.min(), end=value.domain.times.max())
        assert stores[1][0].interval_identity(stores[1][1], start=value.domain.times.min(), end=value.domain.times.max()) is None
        with sqlite3.connect(stores[1][0].meta_path) as db:
            db.execute("INSERT INTO computation_index VALUES(?,?,?,?,?)", saved)
        assert stores[1][0].interval_identity(stores[1][1], start=value.domain.times.min(), end=value.domain.times.max())


def test_large_numerical_preparation_preserves_proofs_in_one_reserve():
    preparation = ExecutionPreparation(4096)
    memo = PreparationMetadataMemo(preparation, "fixture")
    preparation.remember("numerical", "window", lambda: "x" * 2000, len)
    memo.put("original", "y" * 200, 200)
    assert memo.get("original") == "y" * 200
    assert ("numerical", "window") not in preparation._values
    assert preparation.bytes <= preparation.maximum_bytes
    preparation.remember("numerical", "other", lambda: "z" * 2000, len)
    assert memo.get("original") == "y" * 200
    assert preparation.bytes <= preparation.maximum_bytes
    assert preparation.report()["numerical_nonadmissions"] == 1
    memo.put("replacement", "z" * 200, 200)
    assert memo.get("original") is None
    assert memo.get("replacement") == "z" * 200
    preparation.restrict(0)
    assert preparation.bytes == 0


def test_large_metadata_cannot_flush_compact_authenticated_ancestry():
    preparation = ExecutionPreparation(4096)
    memo = PreparationMetadataMemo(preparation, "fixture")
    authority = ("source_bindings", "original", "immutable-content")
    memo.put(authority, ("header", ("original", "parent")), 100)
    for index in range(8):
        memo.put(("manifest", index), "large-body", 350)
        assert memo.get(authority) == ("header", ("original", "parent"))
    assert preparation.bytes <= preparation.maximum_bytes


def test_interval_proofs_keep_authenticated_binding_when_manifest_body_is_evicted(tmp_path, monkeypatch):
    value = retired_source(65)
    graph, store, root = global_graph(tmp_path, negate(value))
    receipt = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    identity = receipt["results"]["main"][root]
    first, last = value.domain.times.min(), value.domain.times.max()
    preparation = ExecutionPreparation(1024 * 1024)
    with preparation_scope(preparation):
        expected = store.interval_identity(identity, start=first, end=last)
        # Model pressure from unrelated large metadata while retaining exact
        # interval evidence and its compact, authenticated owner binding.
        for index in range(3):
            PreparationMetadataMemo(preparation, (store.meta_path, store.artifact_path)).put(
                ("manifest", f"unrelated-{index}"), "body", 100 * 1024)
        monkeypatch.setattr(store, "manifest", lambda *_: pytest.fail("interval proof decoded full manifest again"))
        assert store.interval_identity(identity, start=first, end=last) == expected
        assert preparation.bytes <= preparation.maximum_bytes


@pytest.mark.parametrize("dynamic", [False, True])
def test_proof_only_window_retains_exact_signature_after_coordinate_eviction(dynamic):
    source = retired_source(65).domain
    domain = source if dynamic else Domain(calendar=source.times, universe=source.asset_ids, identity="opaque")
    first, last = domain.times.min(), domain.times.max()
    expected = domain.window(first, last).signature
    preparation = ExecutionPreparation(4096)
    with preparation_scope(preparation):
        assert domain._window_signature(first, last) == expected
        builds = preparation.report()["domain_window_builds"]
        for index in range(10):
            preparation.remember("parent_slab", index, lambda: b"x" * 2500, len)
            assert domain._window_signature(first, last) == expected
        assert preparation.report()["domain_window_builds"] == builds
        other = Domain(calendar=source.times, universe=["foreign"], identity=domain.signature)
        assert other._window_signature(first, last) != expected
        assert preparation.peak_bytes <= preparation.maximum_bytes


def test_lightweight_window_proof_cannot_retain_an_oversized_source_domain():
    import gc
    from weakref import ref
    preparation = ExecutionPreparation(4096)

    def transient_signature():
        days = [date(2000, 1, 1) + timedelta(days=index) for index in range(4096)]
        domain = Domain(calendar=days, universe=["asset"])
        observed = ref(domain)
        assert domain._window_signature(days[0], days[-1])
        return observed

    with preparation_scope(preparation):
        observed = transient_signature()
        gc.collect()
        assert observed() is None
        assert preparation.report()["domain_window_oversize"] == 1
        assert preparation.bytes <= preparation.maximum_bytes


def test_warm_preparation_never_caches_graph_revision_guards(tmp_path):
    value = retired_source(32)
    graph, store, root = global_graph(tmp_path, negate(value))
    receipt = finish(graph.plan_update({"main": {"x": value}}, through=value.domain.times.max()))
    appended = retired_source(33)
    plan = graph.plan_update({"main": {"x": appended}}, through=appended.domain.times.max())
    with preparation_scope(plan._preparation):
        store.describe(receipt["results"]["main"][root])
        graph.sleep(root)
        with pytest.raises(RevisionConflict):
            plan.guard()


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


@pytest.mark.parametrize("planned_cache,worker_cache", [(0, 0), (3, 3), (4, 4),
    (896, 896), (1024, 128), (128, 1024), (1024, 0)])
def test_preparation_scales_with_cache_and_respects_smaller_worker_admission(
        tmp_path, monkeypatch, planned_cache, worker_cache):
    from dataclasses import replace

    value = retired_source(3)
    limits = ResourceLimits(total_threads=6, memory_target_mib=20480, cache_mib=planned_cache)
    with resource_limits(limits):
        graph, store, root = global_graph(tmp_path, negate(value))
        plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
        assert plan._preparation.maximum_bytes == planned_cache // 4 * 1024**2
        selected = replace(limits, cache_mib=worker_cache)
        execute = plan._execute
        calls = []

        def observed(*args, **kwargs):
            admitted = kwargs["limits"]
            reserve = plan._preparation.maximum_bytes // 1024**2
            assert reserve == min(planned_cache // 4, worker_cache // 4)
            assert admitted.cache_mib == worker_cache - worker_cache // 4
            assert reserve + admitted.cache_mib <= worker_cache
            assert admitted.memory_target_mib == limits.memory_target_mib
            calls.append(admitted)
            return execute(*args, **kwargs)

        monkeypatch.setattr(plan, "_execute", observed)
        receipt = finish(plan, limits=selected)
        assert calls and plan.preparation_usage["resident_bytes"] == 0
        repeated = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max())
        assert repeated.complete
        assert finish(repeated)["results"] == receipt["results"]
        assert all(usage["outcome"] == "up_to_date" and usage["computed_blocks"] == 0
            and usage["input_sessions"] == 0
            for usage in repeated.resource_usage["main"].values())
        assert store.read_values(receipt["results"]["main"][root]).height == value.domain.size


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
