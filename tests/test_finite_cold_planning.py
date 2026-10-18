"""Cold planning metadata stays bounded; public numerical/history gates remain.

The controller-only fixtures inject no authority or values into a Graph. Their
purpose is to inspect scalar query counts and conservative admission arithmetic.
Separate actual public Graph cases below compare the unchanged canonical path.
"""
from collections import OrderedDict
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from bagelquant_core import Domain
from bagelquant_core._finite_cold_batch import _ColdFiniteBatches
import bagelquant_core.incremental as incremental
from bagelquant_core.resources import active_resource_limits, resource_limits
from test_finite_cold_batches import (
    LIMITS, NATIVE, assert_preserved, canonical, chain, execute, file_state,
    scientific_inventory,
)
from test_finite_pct_view import bits, indexed_source_proof, source
from test_global_graph import finish


def dense_domain(sessions=320, assets=4):
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(sessions)]
    members = pl.DataFrame({"time": days}).join(pl.DataFrame({
        "asset_id": [f"A{index}" for index in range(assets)]}), how="cross")
    return days, Domain(calendar=days, universe=members.with_columns(pl.lit(True).alias("active")))


def controller(domain, *, limits=LIMITS, parts=(), traces=()):
    """Isolate planner arithmetic, not public eligibility or receipt validation."""
    value = object.__new__(_ColdFiniteBatches)
    value.eligible = True
    value._store = SimpleNamespace(inventory=lambda **_kwargs: ())
    value._node_id = "unit"
    value._parents = {"parent": SimpleNamespace(domain=domain, trace_columns=traces)}
    value._manifests = {"parent": {"partitions": list(parts)}}
    value._keys = {}
    value._schemas = {}
    value._read_starts = OrderedDict()
    longest = max((len(asset.encode("utf-8")) for asset in domain.asset_ids), default=0)
    value._parent_bounds = {"parent": (len(domain.asset_ids), longest)}
    value._output_asset_bytes = longest
    value.schema_reads = 0
    value._domain, value._limits = domain, limits
    value._owned = set()
    value.planning_bytes = 65536 + 4096
    value.planning_limits = replace(limits,
        memory_target_mib=limits.memory_target_mib - 1)
    return value


def huge_unknown_part():
    return {"start": None, "end": None, "rows": 1_000_000}


def test_high_asset_row_rejections_precede_all_start_queries_and_domain_windows(monkeypatch):
    days, domain = dense_domain(96, 4096)
    calls = []
    def forbidden_window(*_args, **_kwargs):
        pytest.fail("planner constructed a temporary Domain")
    monkeypatch.setattr(Domain, "window", forbidden_window)
    limits = replace(LIMITS, batch_rows=250_000)
    with controller(domain, limits=limits) as value:
        selected = value.plan(0, days,
            lambda origin, end: calls.append((origin, end)), 32)
        assert selected is None and calls == [] and not value._read_starts


def test_output_count_is_admitted_membership_not_numeric_or_unique_asset_rows():
    days, domain = dense_domain(96, 17)
    # There is no value frame: every admitted key can be an implicit-null value.
    with controller(domain) as value:
        assert value._output_rows(days[0], days[63]) == 64 * 17
        assert value._output_rows(days[32], days[32]) == 17


def test_rejected_resource_sizes_share_eight_queries_and_next_origin_reuses_seven():
    days, domain = dense_domain()
    calls = []
    def start(origin, end):
        calls.append((origin, end))
        return days[max(0, origin - 9)]
    with controller(domain, parts=[huge_unknown_part()]) as value:
        assert value.plan(0, days, start, 32) is None
        assert calls == [(origin, days[origin + 31]) for origin in range(0, 256, 32)]
        assert len(value._read_starts) == 8
        assert value.plan(32, days, start, 32) is None
        assert calls[-1] == (256, days[287]) and len(calls) == 9
        assert len(value._read_starts) == 8
        assert all(type(answer) is date for answer in value._read_starts.values())


def test_partial_final_end_is_exact_and_is_not_reused_as_a_full_block():
    days, domain = dense_domain(257)
    calls = []
    def start(origin, end):
        calls.append((origin, end))
        return days[max(0, origin - 9)]
    with controller(domain, parts=[huge_unknown_part()]) as value:
        assert value.plan(0, days, start, 32) is None
        assert value.plan(32, days, start, 32) is None
        assert calls[-1] == (256, days[256]) and len(calls) == 9
        assert (256, days[256]) in value._read_starts
        assert (256, days[255]) not in value._read_starts


def test_memo_key_binds_end_and_evicts_before_growing_past_eight():
    days, domain = dense_domain()
    calls = []
    def start(origin, end):
        calls.append((origin, end))
        return end
    with controller(domain) as value:
        for origin in range(0, 256, 32):
            value._numerical_start(origin, days[origin + 31], start)
            assert len(value._read_starts) <= 8
        assert value._numerical_start(0, days[30], start) == days[30]
        assert len(calls) == 9 and len(value._read_starts) == 8
        assert (0, days[31]) not in value._read_starts
        value._numerical_start(0, days[31], start)
        assert len(calls) == 10 and len(value._read_starts) == 8


@pytest.mark.parametrize("answer", [None, pl.DataFrame({"x": [1]})])
def test_nondate_results_are_never_retained(answer):
    days, domain = dense_domain()
    with controller(domain) as value:
        assert value._numerical_start(0, days[31], lambda *_args: answer) is answer
        assert not value._read_starts


@pytest.mark.parametrize("error", [RuntimeError("scalar"), ValueError("invalid"), KeyboardInterrupt()])
def test_scalar_error_propagates_without_caching_or_another_attempt(error):
    days, domain = dense_domain()
    calls = []
    def broken(origin, end):
        calls.append((origin, end))
        raise error
    value = controller(domain)
    with pytest.raises(type(error)) as captured:
        with value:
            value.plan(0, days, broken, 32)
    assert captured.value is error and calls == [(0, days[31])]
    assert not value._read_starts and not value._parent_bounds
    assert value._domain is value._parents is value._store is None


def test_close_releases_only_scalar_metadata_and_source_authority_is_unchanged():
    days, domain = dense_domain()
    identity = domain.signature
    value = controller(domain)
    with value:
        assert value._numerical_start(0, days[31], lambda *_args: days[0]) == days[0]
        assert value._read_starts
    assert not value.eligible and not value._read_starts and not value._parent_bounds
    assert value._output_asset_bytes == 0 and domain.signature == identity


@pytest.mark.parametrize("traces", [(), ("available_date", "observed_at")])
def test_global_axes_and_physical_rows_are_conservative_without_domain_construction(monkeypatch, traces):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(128)]
    names = ["A", *["长资产" + str(i) for i in range(16)]]
    keys = [(day, asset) for i, day in enumerate(days) for asset in (["A"] if i < 64 else names)]
    domain = Domain(calendar=days, universe=pl.DataFrame(keys,
        schema={"time": pl.Date, "asset_id": pl.String}, orient="row").with_columns(pl.lit(True).alias("active")))
    monkeypatch.setattr(Domain, "window",
        lambda *_args, **_kwargs: pytest.fail("selection constructed a Domain"))
    parts = [{"start": None, "end": None, "rows": 1500},
        {"start": str(days[96]), "end": str(days[-1]), "rows": 1_000_000}]
    limits = replace(LIMITS, batch_rows=512)
    with controller(domain, limits=limits, parts=parts, traces=traces) as value:
        with resource_limits(limits):
            batch = value.plan(0, days, lambda *_args: days[0], 32)
            assert active_resource_limits() is limits
        assert batch is not None and batch.stop == 64
        longest = max(len(name.encode("utf-8")) for name in names)
        raw = (value.planning_bytes + 8 * 64 * (64 + longest + 16)
            + 8 * 1500 * (64 + longest + 8 * len(traces)))
        assert batch.raw_bytes == raw
        assert batch.reserve_mib == (raw + 1024**2 - 1) // 1024**2
        assert batch.limits == replace(limits,
            memory_target_mib=limits.memory_target_mib - batch.reserve_mib)
        # The selected window contains only A; using all17 assets overstates,
        # rather than understates, the old64-row dense rectangle.
        assert 64 * len(names) > 64


@pytest.mark.parametrize("bounds", [(None, None), (None, "end"), ("start", None)])
def test_unknown_physical_bounds_are_never_pruned(bounds):
    days, domain = dense_domain()
    part = {"start": None if bounds[0] is None else str(days[0]),
        "end": None if bounds[1] is None else str(days[-1]), "rows": 1_000_000}
    with controller(domain, parts=[part]) as value:
        assert value.plan(0, days, lambda *_args: days[0], 32) is None


def test_global_axis_upper_bound_may_decline_but_does_not_underreserve():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(96)]
    keys = [(day, "A") for day in days]
    keys.extend((days[-1], f"LATER{index}") for index in range(100))
    domain = Domain(calendar=days, universe=pl.DataFrame(keys,
        schema={"time": pl.Date, "asset_id": pl.String}, orient="row").with_columns(pl.lit(True).alias("active")))
    with controller(domain, limits=replace(LIMITS, batch_rows=100)) as value:
        # First64 rows fit, but the conservative64x101 parent rectangle exceeds
        # eight times100; no reserve or fabricated local asset count is used.
        assert value.plan(0, days, lambda *_args: days[0], 32) is None


def test_count_failure_is_not_swallowed_and_never_reaches_a_history_query(monkeypatch):
    days, domain = dense_domain()
    error, calls = ValueError("metadata count"), []
    value = controller(domain)
    def broken(*_args):
        raise error
    monkeypatch.setattr(value, "_output_rows", broken)
    with pytest.raises(ValueError) as captured:
        with value:
            value.plan(0, days, lambda *args: calls.append(args), 32)
    assert captured.value is error and not calls and not value._read_starts


def test_reentry_disproves_monotonic_first_origin_and_keeps_exact_read_starts():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    keys = [(day, "DENSE") for day in days] + [(days[0], "RETURN"), (days[90], "RETURN")]
    domain = Domain(calendar=days, universe=pl.DataFrame(keys,
        schema={"time": pl.Date, "asset_id": pl.String}, orient="row").with_columns(pl.lit(True).alias("active")))
    def original(origin, end):
        return incremental._numerical_read_start(domain, days[origin], end, 9)
    with controller(domain) as value:
        first = value._numerical_start(32, days[63], original)
        later = value._numerical_start(64, days[95], original)
        assert first == original(32, days[63]) == days[23]
        assert later == original(64, days[95]) == days[0] < first
        assert min(first, later) == days[0]


@NATIVE
@pytest.mark.parametrize("kind", ["pct", "smooth"])
def test_public_sparse_cold_matches_canonical_bits_proofs_and_noop_history(tmp_path, monkeypatch, kind):
    x = source(640)
    limits = replace(LIMITS, batch_rows=320)
    expression = chain(x, kind)
    graph, store, root, plan, result = execute(tmp_path / "future", x,
        expression=expression, limits=limits)
    with canonical(monkeypatch):
        _, reference, reference_root, reference_plan, expected = execute(tmp_path / "reference", x,
            expression=expression, limits=limits)
    assert root == reference_root and result["results"] == expected["results"]
    assert scientific_inventory(store) == scientific_inventory(reference)
    assert bits(store.read(result["results"]["main"][root]).panel.collect(dense=False, include_traces=True)) == bits(
        reference.read(expected["results"]["main"][root]).panel.collect(dense=False, include_traces=True))
    actual, prior = plan.resource_usage["main"][root], reference_plan.resource_usage["main"][root]
    assert set(actual) == set(prior) == {"canonical_sessions", "computed_blocks", "reused_blocks",
        "input_sessions", "kernel_sessions", "computed_output_sessions"}
    assert actual["computed_blocks"] == prior["computed_blocks"] == 20
    assert actual["computed_output_sessions"] == prior["computed_output_sessions"] == 640
    assert actual["reused_blocks"] == prior["reused_blocks"] == 0
    assert actual["input_sessions"] < prior["input_sessions"]
    assert actual["kernel_sessions"] < prior["kernel_sessions"]
    before = file_state(store)
    with monkeypatch.context() as patch:
        patch.setattr(_ColdFiniteBatches, "_output_rows",
            lambda *_args: pytest.fail("no-op performed batch planning"))
        patch.setattr(_ColdFiniteBatches, "_numerical_start",
            lambda *_args: pytest.fail("no-op queried historical starts"))
        repeated = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(),
            input_proof=indexed_source_proof(x))
        assert repeated.complete
        assert finish(repeated, limits=limits)["results"] == result["results"]
    assert file_state(store) == before


@NATIVE
@pytest.mark.parametrize("change", ["append", "repair"])
def test_retained_history_never_enters_new_planner_and_stays_bounded(tmp_path, monkeypatch, change):
    old = source(96)
    graph, store, root, _, _ = execute(tmp_path / "history", old)
    before = file_state(store)
    x = source(160) if change == "append" else source(96, revision=70)
    graph.merge(chain(x))
    with monkeypatch.context() as patch:
        patch.setattr(_ColdFiniteBatches, "_output_rows",
            lambda *_args: pytest.fail("retained history entered cold row planning"))
        patch.setattr(_ColdFiniteBatches, "_numerical_start",
            lambda *_args: pytest.fail("retained history entered cold scalar memo"))
        plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(),
            input_proof=indexed_source_proof(x))
        actual = finish(plan, limits=LIMITS)
    _, fresh, fresh_root, _, expected = execute(tmp_path / "fresh", x)
    assert root == fresh_root and actual["results"] == expected["results"]
    assert bits(store.read(actual["results"]["main"][root]).panel.collect(dense=False, include_traces=True)) == bits(
        fresh.read(expected["results"]["main"][root]).panel.collect(dense=False, include_traces=True))
    assert plan.resource_usage["main"][root]["reused_blocks"] > 0
    assert plan.resource_usage["main"][root]["computed_blocks"] <= 2
    assert_preserved(store, before)


@NATIVE
def test_durable_cancellation_closes_scalars_and_retry_keeps_original_frontier(tmp_path, monkeypatch):
    from bagelquant_core import CoreStore
    from test_global_graph import global_graph

    x = source(96)
    graph, store, root = global_graph(tmp_path / "interrupted", chain(x))
    plan = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(),
        input_proof=indexed_source_proof(x))
    controllers, committed = [], []
    factory, publish = incremental._ColdFiniteBatches, CoreStore.publish
    def retained(*args, **kwargs):
        value = factory(*args, **kwargs)
        controllers.append(value)
        return value
    def durable(owner, value, **kwargs):
        result = publish(owner, value, **kwargs)
        if value.key.node_id == root and kwargs.get("value_references") is None:
            committed.append(value.key.identity)
        return result
    def cancel():
        if committed:
            raise RuntimeError("stop after committed block")
    with monkeypatch.context() as patch:
        patch.setattr(incremental, "_ColdFiniteBatches", retained)
        patch.setattr(CoreStore, "publish", durable)
        with pytest.raises(RuntimeError, match="committed block"):
            finish(plan, limits=LIMITS, check_canceled=cancel)
    assert len(committed) == 1 and store.describe(committed[0]) is not None
    assert controllers and all(not value._read_starts and not value._parent_bounds
        and value._domain is value._parents is None for value in controllers)
    before = file_state(store)
    with monkeypatch.context() as patch:
        patch.setattr(_ColdFiniteBatches, "_output_rows",
            lambda *_args: pytest.fail("retry entered cold row planning"))
        patch.setattr(_ColdFiniteBatches, "_numerical_start",
            lambda *_args: pytest.fail("retry entered cold scalar memo"))
        retry = graph.plan_update({"main": {"x": x}}, through=x.domain.times.max(),
            input_proof=indexed_source_proof(x))
        actual = finish(retry, limits=LIMITS)
    assert retry.resource_usage["main"][root]["reused_blocks"] == 1
    assert retry.resource_usage["main"][root]["computed_blocks"] == 2
    assert_preserved(store, before)
    assert store.manifest(actual["results"]["main"][root])["value_sources"][0]["identity"] == committed[0]
    _, fresh, fresh_root, _, expected = execute(tmp_path / "fresh", x)
    assert root == fresh_root and actual["results"] == expected["results"]
    assert bits(store.read(actual["results"]["main"][root]).panel.collect(dense=False, include_traces=True)) == bits(
        fresh.read(expected["results"]["main"][root]).panel.collect(dense=False, include_traces=True))
