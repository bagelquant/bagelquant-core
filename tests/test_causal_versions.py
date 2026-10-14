from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import CausalInputWindow, CoreStore, Domain, ExecutionRuntime, Graph, Node
from bagelquant_core.hashing import hash_dataframe
from bagelquant_core.operator import mad_zscore, rolling_mean, sub
from test_global_graph import finish


def fixture(count=70, *, late=False, ipo=False):
    days = [date(2024, 1, 1)+timedelta(days=i) for i in range(count)]
    rows = [{"time": day, "asset_id": asset, "value": float((index*3+j*j)%19),
        "known_at": day, "revision": index*10+j}
        for index, day in enumerate(days) for j, asset in enumerate("abcde")]
    if late:
        rows.extend([{"time": days[49], "asset_id": "a", "value": 97., "known_at": days[52], "revision": 9999},
            {"time": days[54], "asset_id": "b", "value": None, "known_at": days[55], "revision": 10000}])
    if ipo:
        rows.extend({"time": day, "asset_id": "ipo", "value": 1., "known_at": day, "revision": 20000+index}
                    for index, day in enumerate(days[70:]))
    frame = pl.DataFrame(rows, schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64,
        "known_at": pl.Date, "revision": pl.UInt64})
    membership = frame.select("time", "asset_id").unique().with_columns(pl.lit(True).alias("active"))
    domain = Domain(calendar=days, universe=membership)
    value = Node.from_domain(pl.DataFrame(schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64}),
        domain, source_key="x", identity=hash_dataframe(frame))
    return frame, value


def setup(tmp_path, value):
    store = CoreStore(tmp_path/"core.sqlite", tmp_path/"values")
    graph = Graph(store=store, graph_id="global")
    expression = mad_zscore(rolling_mean(rolling_mean(value, window=3), window=2))
    root = next(iter(graph.merge(expression).outputs.values()))
    graph.register_context("main", anchor="2024-01-01", information_cutoff=str(value.domain.times.max()))
    return graph, store, root


def plan(graph, versions, value, reads, *, definition=None):
    def proof(context, source, first, last, cutoff):
        return hash_dataframe(versions.filter(pl.col("time").is_between(first, last) & (pl.col("known_at") <= cutoff)))
    def load(context, source, first, last, cutoff):
        reads.append((first, last, cutoff))
        return CausalInputWindow(versions.filter(pl.col("time").is_between(first, last) &
            (pl.col("known_at") <= cutoff)), value.identity)
    return graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(),
        causal_inputs={"x"}, input_version_loader=load, input_scope_proof=proof,
        input_definition_identities=None if definition is None else {"main": {"x": definition}})


def oracle(versions, domain):
    days = domain.times.to_list()
    assets = domain.asset_ids.to_list()
    output = []
    for index, day in enumerate(days):
        current = versions.filter(pl.col("known_at") <= day).sort("known_at", "revision").unique(
            ["time", "asset_id"], keep="last")
        lookup = {(row["time"], row["asset_id"]): row["value"] for row in current.iter_rows(named=True)}
        values = []
        for asset in assets:
            outer = []
            for end in range(index-1, index+1):
                raw = [lookup.get((days[position], asset)) for position in range(max(0, end-2), end+1)] if end >= 0 else []
                outer.append(float(np.mean(raw)) if len(raw) == 3 and all(value is not None for value in raw) else None)
            values.append(float(np.mean(outer)) if all(value is not None for value in outer) else np.nan)
        # Independent legacy population z-score; this sample has no outliers.
        x = np.asarray(values)
        valid = np.isfinite(x)
        result = np.full(x.shape, np.nan)
        if valid.any():
            sample = x[valid].copy()
            median = np.median(sample)
            mad = np.median(np.abs(sample-median))
            if mad > 0:
                sample = np.clip(sample, median-5*1.4826*mad, median+5*1.4826*mad)
            if sample.std(ddof=0) > 1e-12:
                result[valid] = (sample-sample.mean())/sample.std(ddof=0)
        output.extend({"time": day, "asset_id": asset, "value": float(value) if np.isfinite(value) else None}
                      for asset, value in zip(assets, result, strict=True))
    return domain.apply_membership_lazy(pl.DataFrame(output, schema={"time": pl.Date,
        "asset_id": pl.String, "value": pl.Float64}).lazy()).collect().sort("time", "asset_id")


def test_late_revision_reselects_nested_halo_without_rewriting_earlier_outputs(tmp_path):
    versions, value = fixture(late=True)
    graph, store, root = setup(tmp_path, value)
    reads = []
    update = plan(graph, versions, value, reads)
    receipt = finish(update)
    actual = store.read(receipt["results"]["main"][root]).panel.collect(dense=False)
    assert_frame_equal(actual, oracle(versions, value.domain), check_exact=False, abs_tol=1e-12)
    assert len(reads) == 3  # One acquisition per canonical block, shared by all nodes/segments.
    assert all(reads[index][1] < reads[index+1][0] for index in range(len(reads)-1))
    assert update.resource_usage["main"]


def test_shared_source_and_rolling_branches_bind_one_consumed_domain(tmp_path):
    versions, value = fixture(75, late=True)
    store = CoreStore(tmp_path/"core.sqlite", tmp_path/"values")
    graph = Graph(store=store, graph_id="global")
    difference = sub(rolling_mean(value, window=3), value)
    root = next(iter(graph.merge(difference).outputs.values()))
    graph.register_context("main", anchor="2024-01-01", information_cutoff=str(value.domain.times.max()))
    reads = []
    update = plan(graph, versions, value, reads)
    receipt = finish(update)
    expected = []
    days = value.domain.times.to_list()
    for index, day in enumerate(days):
        current = versions.filter(pl.col("known_at") <= day).sort("known_at", "revision").unique(
            ["time", "asset_id"], keep="last")
        lookup = {(row["time"], row["asset_id"]): row["value"] for row in current.iter_rows(named=True)}
        for asset in value.domain.asset_ids:
            raw = [lookup.get((previous, asset)) for previous in days[max(0, index-2):index+1]]
            actual = (float(np.mean(raw))-raw[-1] if len(raw) == 3 and all(item is not None for item in raw)
                else None)
            expected.append({"time": day, "asset_id": asset, "value": actual})
    assert_frame_equal(store.read(receipt["results"]["main"][root]).panel.collect(dense=False),
        pl.DataFrame(expected, schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64}),
        check_exact=False, abs_tol=1e-12)
    assert len(reads) == 3
    assert all(reads[index][1] < reads[index+1][0] for index in range(len(reads)-1))


def test_append_ipo_preserves_blocks_and_repeat_is_metadata_only(tmp_path, monkeypatch):
    original, first = fixture(65)
    graph, store, root = setup(tmp_path, first)
    before = finish(plan(graph, original, first, []))
    old = store.read(before["results"]["main"][root]).panel.collect(dense=False)
    versions, current = fixture(75, late=True, ipo=True)
    graph.register_context("main", anchor="2024-01-01", information_cutoff=str(current.domain.times.max()))
    reads = []
    update = plan(graph, versions, current, reads)
    receipt = finish(update)
    actual = store.read(receipt["results"]["main"][root]).panel.collect(dense=False)
    assert_frame_equal(actual, oracle(versions, current.domain), check_exact=False, abs_tol=1e-12)
    assert_frame_equal(actual.filter(pl.col("time") < date(2024, 2, 22)), old.filter(pl.col("time") < date(2024, 2, 22)), check_exact=True)
    assert update.resource_usage["main"][root]["reused_blocks"] >= 1
    def forbidden(*args, **kwargs):
        pytest.fail("unchanged repeat read values or ran a kernel")
    monkeypatch.setattr(store, "lookup", forbidden)
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    repeat = graph.plan_update({"main": {"x": current}}, through=current.domain.times.max(),
        causal_inputs={"x"}, input_version_loader=forbidden, input_scope_proof=forbidden)
    assert repeat.complete
    assert repeat.input_proof_windows("main").is_empty()
    assert finish(repeat)["results"] == receipt["results"]


def test_cancel_reuses_committed_causal_blocks(tmp_path):
    versions, value = fixture(late=True)
    graph, store, root = setup(tmp_path, value)
    interrupted = plan(graph, versions, value, [])
    def cancel():
        if sum(store.manifest(item["identity"])["through"] == "2024-02-01" for item in store.inventory()) == 3:
            raise RuntimeError("canceled")
    with pytest.raises(RuntimeError, match="canceled"):
        interrupted.execute("main", interrupted.ready("main")[0], check_canceled=cancel)
    prior = {item["identity"] for item in store.inventory()}
    reads = []
    restarted = plan(graph, versions, value, reads)
    result = finish(restarted)
    assert prior <= {item["identity"] for item in store.inventory()}
    assert all(last != date(2024, 2, 1) for first, last, cutoff in reads)
    assert_frame_equal(store.read(result["results"]["main"][root]).panel.collect(dense=False),
        oracle(versions, value.domain), check_exact=False, abs_tol=1e-12)


def test_late_first_natural_key_has_separate_source_coverage_and_output_membership(tmp_path):
    versions, original = fixture(75)
    days = original.domain.times.to_list()
    versions = versions.with_columns(pl.when((pl.col("time") == days[49]) & (pl.col("asset_id") == "a"))
        .then(pl.lit(days[52])).otherwise(pl.col("known_at")).alias("known_at"))
    source_domain = Domain(calendar=days, universe=original.domain.asset_ids)
    source = Node.from_domain(pl.DataFrame(schema={"time": pl.Date,"asset_id": pl.String,"value": pl.Float64}),
        source_domain, source_key="x", identity=hash_dataframe(versions))
    output = Domain(calendar=days, universe=versions.filter(pl.col("known_at") <= pl.col("time")).select(
        "time", "asset_id").unique().with_columns(pl.lit(True).alias("active")))
    graph, store, root = setup(tmp_path, source)
    reads = []
    def load(context, key, first, last, cutoff):
        reads.append((first,last))
        return CausalInputWindow(versions.filter(pl.col("time").is_between(first,last)&(pl.col("known_at")<=cutoff)),source.identity)
    def proof(context, key, first,last,cutoff):
        return hash_dataframe(versions.filter(pl.col("time").is_between(first,last)&(pl.col("known_at")<=cutoff)))
    update = graph.plan_update({"main":{"x":source}},through=days[-1],causal_inputs={"x"},
        input_version_loader=load,input_scope_proof=proof,causal_output_domains={"main":output})
    result = finish(update)
    actual = store.read(result["results"]["main"][root]).panel.collect(dense=False)
    assert_frame_equal(actual,oracle(versions,output),check_exact=False,abs_tol=1e-12)
    assert actual.filter((pl.col("time")==days[49])&(pl.col("asset_id")=="a")).is_empty()


def test_unproven_entire_history_blocks_before_source_or_kernel(tmp_path, monkeypatch):
    versions, value = fixture()
    graph, store, root = setup(tmp_path, value)
    finish(plan(graph, versions, value, []))
    changed = Node.from_domain(pl.DataFrame(schema={"time":pl.Date,"asset_id":pl.String,"value":pl.Float64}),
        value.domain,source_key="x",identity="new-source-authority")
    def forbidden(*args, **kwargs):
        pytest.fail("unproven completed history consumed values")
    update = graph.plan_update({"main":{"x":changed}},through=value.domain.times.max(),causal_inputs={"x"},
        input_version_loader=forbidden,input_scope_proof=lambda *args:"unrelated-proof-namespace")
    monkeypatch.setattr(ExecutionRuntime,"run",forbidden)
    with pytest.raises(ValueError,match="no proven retained interval"):
        update.execute("main",update.ready("main")[0])


def test_relevant_upstream_implementation_change_creates_new_causal_authority(tmp_path, monkeypatch):
    versions, value = fixture()
    graph, store, root = setup(tmp_path,value)
    before=finish(plan(graph,versions,value,[]))
    monkeypatch.setattr(rolling_mean,"version","new-numerical-implementation")
    reads=[]
    update=plan(graph,versions,value,reads)
    after=finish(update)
    assert before["results"]!=after["results"]
    assert len(reads)==3
    assert_frame_equal(store.read(after["results"]["main"][root]).panel.collect(dense=False),
        oracle(versions,value.domain),check_exact=False,abs_tol=1e-12)


def test_source_definition_change_rebuilds_and_retains_original_authority(tmp_path, monkeypatch):
    versions, value = fixture()
    graph, store, root = setup(tmp_path, value)
    before = finish(plan(graph, versions, value, [], definition="source-implementation-v1"))
    original = store.read(before["results"]["main"][root]).panel.collect(dense=False)
    revised = versions.with_columns((pl.col("value") + pl.col("revision") % 7).alias("value"))
    reads = []
    # Keep the same placeholder content identity: definition authority itself
    # must invalidate the compact completion and full result key.
    update = plan(graph, revised, value, reads, definition="source-implementation-v2")
    after = finish(update)
    assert before["results"] != after["results"]
    assert len(reads) == 3
    assert_frame_equal(store.read(after["results"]["main"][root]).panel.collect(dense=False),
        oracle(revised, value.domain), check_exact=False, abs_tol=1e-12)
    assert_frame_equal(store.read(before["results"]["main"][root]).panel.collect(dense=False), original)
    def forbidden(*args, **kwargs):
        pytest.fail("unchanged source definition consumed values or ran a kernel")
    monkeypatch.setattr(store, "lookup", forbidden)
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    repeat = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(),
        causal_inputs={"x"}, input_version_loader=forbidden, input_scope_proof=forbidden,
        input_definition_identities={"main": {"x": "source-implementation-v2"}})
    assert repeat.complete
    assert finish(repeat)["results"] == after["results"]


def test_stable_source_definition_append_reuses_history_and_unknown_proofs_still_block(tmp_path):
    versions, value = fixture(65)
    graph, store, root = setup(tmp_path, value)
    before = finish(plan(graph, versions, value, [], definition="source-implementation-v1"))
    current_versions, current = fixture(75)
    graph.register_context("main", anchor="2024-01-01", information_cutoff=str(current.domain.times.max()))
    reads = []
    update = plan(graph, current_versions, current, reads, definition="source-implementation-v1")
    after = finish(update)
    assert update.resource_usage["main"][root]["computed_output_sessions"] == 10
    assert update.resource_usage["main"][root]["reused_blocks"] >= 2
    assert all(first > value.domain.times.max() or (last-first).days < 32 for first, last, _ in reads)
    assert_frame_equal(store.read(after["results"]["main"][root]).panel.collect(dense=False),
        oracle(current_versions, current.domain), check_exact=False, abs_tol=1e-12)
    assert store.describe(before["results"]["main"][root]) is not None
    changed = Node.from_domain(pl.DataFrame(schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64}),
        current.domain, source_key="x", identity="unproven-content-namespace")
    def forbidden(*args, **kwargs):
        pytest.fail("unchanged source definition silently cold-replayed unknown history")
    blocked = graph.plan_update({"main": {"x": changed}}, through=current.domain.times.max(), causal_inputs={"x"},
        input_version_loader=forbidden, input_scope_proof=lambda *args: "unrelated-proof-namespace",
        input_definition_identities={"main": {"x": "source-implementation-v1"}})
    with pytest.raises(ValueError, match="no proven retained interval"):
        blocked.execute("main", blocked.ready("main")[0])


def test_source_definition_authority_is_scoped_to_actual_ancestors(tmp_path):
    versions, value = fixture(40)
    other = Node.from_domain(value.collect(dense=False), value.domain, source_key="y", identity=value.identity)
    store = CoreStore(tmp_path / "core.sqlite", tmp_path / "values")
    graph = Graph(store=store, graph_id="global")
    left, right = mad_zscore(value), mad_zscore(other)
    roots = graph.merge(Graph(outputs=[left, right])).outputs
    graph.register_context("main", anchor="2024-01-01", information_cutoff=str(value.domain.times.max()))
    loaded = []
    def proof(context, key, first, last, cutoff):
        return hash_dataframe(versions.filter(pl.col("time").is_between(first, last) & (pl.col("known_at") <= cutoff)))
    def load(context, key, first, last, cutoff):
        loaded.append(key)
        return CausalInputWindow(versions.filter(pl.col("time").is_between(first, last)), value.identity)
    def admitted(definition):
        return graph.plan_update({"main": {"x": value, "y": other}}, through=value.domain.times.max(),
            causal_inputs={"x", "y"}, input_version_loader=load, input_scope_proof=proof,
            input_definition_identities={"main": {"x": "stable-x", "y": definition}})
    before = finish(admitted("old-y"))
    loaded.clear()
    after = finish(admitted("new-y"))
    assert set(loaded) == {"y"}
    assert before["results"]["main"][roots[left.name]] == after["results"]["main"][roots[left.name]]
    assert before["results"]["main"][roots[right.name]] != after["results"]["main"][roots[right.name]]


@pytest.mark.parametrize("definitions", [{}, {"wrong": {"x": "v1"}}, {"main": {}},
    {"main": {"x": ""}}, {"main": {"x": None}}])
def test_source_definition_boundary_rejects_incomplete_or_invalid_authority(tmp_path, definitions):
    versions, value = fixture()
    graph, _, _ = setup(tmp_path, value)
    with pytest.raises(ValueError, match="input definition identities"):
        graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), causal_inputs={"x"},
            input_version_loader=lambda *args: None, input_scope_proof=lambda *args: "proof",
            input_definition_identities=definitions)
