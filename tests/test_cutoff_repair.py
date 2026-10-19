"""Source-admitted final-session repair retains every earlier numerical result."""
from copy import deepcopy
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, ExecutionRuntime, Graph, Node, RevisionConflict
from bagelquant_core.resources import ResourceLimits, active_resource_limits
from bagelquant_core.hashing import hash_mapping
from test_cutoff_proof import prepared
from test_global_graph import finish, global_graph
from test_proof_maintenance import projection


def repair_case(tmp_path, *, count=65, dynamic=False):
    before, _, graph, store, root, update, kwargs = prepared(tmp_path, count=count, dynamic=dynamic)
    end = before.domain.times.max()
    frame = before.collect(include_traces=True).with_columns(
        pl.when((pl.col("time") == end) & (pl.col("asset_id") == "a")).then(31.0).otherwise(pl.col("value")).alias("value"))
    after = Node.from_domain(frame, before.domain, identity="changed-tail", trace_columns=before.trace_columns,
        trace_identity="changed-tail-traces")
    kwargs.update(mode="cutoff_tail_repair.v1", repair_from=end, replacement_inputs={"x": after})
    return before, after, graph, store, root, update, kwargs


def admitted(graph, store, root, update, after, kwargs):
    proof = store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root], **kwargs)
    graph.register_context("main", **kwargs["replacement_context"])
    def window(_context, _input, start, end):
        return Node.from_domain(after.lazy(include_traces=True).filter(pl.col("time").is_between(start, end)),
            after.domain.window(start, end), identity=after.identity, trace_columns=after.trace_columns, trace_identity=after.trace_identity)
    plan = graph.plan_update({"main": {"x": after}}, through=after.domain.times.max(), input_window_loader=window,
        input_proof=lambda _c, _k, first, last: after.interval_identity(first, last))
    store.apply_input_proof_rebind(proof, update_plan=plan)
    return plan, proof


@pytest.mark.parametrize("count,dynamic", [(40, False), (65, True), (270, False)])
def test_repair_one_output_session_matches_full_and_preserves_prefix(tmp_path, monkeypatch, count, dynamic):
    before, after, graph, store, root, update, kwargs = repair_case(tmp_path, count=count, dynamic=dynamic)
    originals = {identity: store.manifest(identity) for identity in update["results"]["main"].values()}
    files = {part["$file"]: store._file(part["$file"]).read_bytes()
        for item in originals.values() for part in item["partitions"]}
    plan, proof = admitted(graph, store, root, update, after, kwargs)
    import bagelquant_core.finite_work as work
    original_run = work._run_finite_outputs
    calls = []
    def bounded(spec, nodes, scoped, parameters, output_start, guard):
        assert output_start == after.domain.times.max()
        calls.append((nodes, {key: len(value.domain.times) for key, value in scoped.items()}))
        return original_run(spec, nodes, scoped, parameters, output_start, guard)
    monkeypatch.setattr(work, "_run_finite_outputs", bounded)
    monkeypatch.setattr("bagelquant_core.regions._run_finite_outputs", bounded)
    result = finish(plan)
    assert len(calls) == 3
    assert all(usage["computed_output_sessions"] == 1 for usage in plan.resource_usage["main"].values())
    assert sorted(usage["kernel_sessions"] for usage in plan.resource_usage["main"].values()) == [1, 1, 2, min(241, count)]
    assert all(store.manifest(identity) == item for identity, item in originals.items())
    assert all(store._file(path).read_bytes() == content for path, content in files.items())
    for node, identity in result["results"]["main"].items():
        actual = store.read_values(identity, end=before.domain.times[-2], include_traces=True)
        expected = store.read_values(update["results"]["main"][node], end=before.domain.times[-2], include_traces=True)
        assert_frame_equal(actual, expected, check_exact=True)
    store.verify_update(result["identity"])
    with monkeypatch.context() as patch:
        patch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("repeat ran numerical work"))
        repeat = finish(graph.plan_update({"main": {"x": after}}, through=after.domain.times.max()))
        assert repeat["results"] == result["results"]
    monkeypatch.undo()
    fresh_graph, fresh_store, fresh_root = global_graph(tmp_path / "fresh",
        Graph.from_logical_spec(graph.snapshot()["spec"], inputs={"x": after}))
    fresh_graph.register_context("main", **kwargs["replacement_context"])
    reference = finish(fresh_graph.plan_update({"main": {"x": after}}, through=after.domain.times.max()))
    assert_frame_equal(store.read_values(result["results"]["main"][root], include_traces=True),
        fresh_store.read_values(reference["results"]["main"][fresh_root], include_traces=True), check_exact=True)
    assert proof["repair"]["prefix_through"] == str(before.domain.times[-2])


@pytest.mark.parametrize("problem", ["earlier_change", "unknown", "wrong_date", "wrong_context", "forged"])
def test_invalid_repair_does_not_authorize_kernels(tmp_path, monkeypatch, problem):
    before, after, graph, store, root, update, kwargs = repair_case(tmp_path)
    monkeypatch.setattr(ExecutionRuntime, "run", lambda *_a, **_k: pytest.fail("invalid repair ran a kernel"))
    if problem == "earlier_change":
        frame = after.collect(include_traces=True).with_columns(
            pl.when(pl.col("time") == before.domain.times[3]).then(-21.0).otherwise(pl.col("value")).alias("value"))
        kwargs["replacement_inputs"] = {"x": Node.from_domain(frame, after.domain, trace_columns=after.trace_columns)}
    elif problem == "unknown":
        kwargs["replacement_projection"] = lambda *_: None
    elif problem == "wrong_date":
        kwargs["repair_from"] = before.domain.times[-2]
    if problem in {"wrong_context", "forged"}:
        plan, proof = admitted(graph, store, root, update, after, kwargs)
        if problem == "wrong_context":
            graph.register_context("main", anchor="2024-01-02", information_cutoff=after.domain.times.max())
        else:
            proof = deepcopy(proof)
            proof["repair"]["history"][root] = 5
            proof["identity"] = hash_mapping({key: value for key, value in proof.items() if key != "identity"})
        with pytest.raises((ValueError, RevisionConflict)):
            store.apply_input_proof_rebind(proof, update_plan=plan)
    else:
        with pytest.raises(ValueError):
            store.plan_input_proof_rebind(graph.snapshot()["spec"], update["results"]["main"][root], **kwargs)


def test_repair_cancellation_retains_prefix_and_resumes(tmp_path, monkeypatch):
    _, after, graph, store, root, update, kwargs = repair_case(tmp_path, count=40)
    plan, proof = admitted(graph, store, root, update, after, kwargs)
    first = plan.ready("main")[0]
    plan.execute("main", first)
    first_identity = plan.results["main"][first]
    def cancel():
        raise RuntimeError("canceled")
    with pytest.raises(RuntimeError, match="canceled"):
        finish(plan, check_canceled=cancel)
    retry, _ = admitted(graph, store, root, update, after, kwargs)
    result = finish(retry)
    assert result["results"]["main"][first] == first_identity
    assert all(usage["computed_output_sessions"] <= 1 for usage in retry.resource_usage["main"].values())
    with store.transaction() as connection:
        connection.execute("DELETE FROM input_proof_rebinding_plans WHERE identity=?", (proof["identity"],))
    with pytest.raises(ValueError, match="admission"):
        store.describe(result["results"]["main"][root])


@pytest.mark.parametrize("expression", ["output = bfill(x, limit=2)", "output = ffill(x)", "output = rolling_mean(x, window=3)"])
def test_future_or_unbounded_support_is_not_admitted(tmp_path, expression):
    before, after, _, _, _, _, _ = repair_case(tmp_path / "source")
    graph, store, root = global_graph(tmp_path / "unsupported", Graph.from_dsl(expression, inputs={"x": before}))
    end = before.domain.times.max()
    graph.register_context("main", anchor="2024-01-01", information_cutoff=end + timedelta(days=2))
    old = graph.snapshot()["contexts"]["main"]
    result = finish(graph.plan_update({"main": {"x": before}}, through=end))
    with pytest.raises(ValueError, match="bounded backward support"):
        store.plan_input_proof_rebind(graph.snapshot()["spec"], result["results"]["main"][root],
            mode="cutoff_tail_repair.v1", repair_from=end, original_update=result["identity"], context_id="main",
            original_context=old, replacement_context={**old, "information_cutoff": str(end)},
            original_inputs={"x": before}, replacement_inputs={"x": after},
            original_projection=projection, replacement_projection=projection,
            report_id="unsupported", report_digest="a" * 64)


def test_sparse_reentry_nulls_and_limited_fill_use_actual_predecessors(tmp_path):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(400)]
    rows = [(day, asset, (0. if asset == "d" else 2. if i > 260 else 1.), day)
        for i, day in enumerate(days) for asset in ("a", "b", "c", "d")
        if asset in {"a", "d"} or asset == "b" and (i < 5 or i == 399) or asset == "c" and i == 399]
    frame = pl.DataFrame(rows, schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64,
        "available_date": pl.Date}, orient="row").with_columns(
        pl.when((pl.col("asset_id") == "d") & (pl.col("time") == days[398])).then(float("nan"))
        .otherwise(pl.col("value")).alias("value"))
    domain = Domain(calendar=days, universe=frame.select("time", "asset_id").with_columns(pl.lit(True).alias("active")))
    before = Node.from_domain(frame, domain, trace_columns=("available_date",), source_key="x")
    after = Node.from_domain(frame.with_columns(pl.when((pl.col("asset_id") == "b") & (pl.col("time") == days[-1]))
        .then(4.).otherwise(pl.col("value")).alias("value")), domain, trace_columns=("available_date",))
    expression = Graph.from_dsl("p = pct_change_from_last_change(x)\nf = ffill(p, limit=240)\noutput = canonicalize_values(rankpct(f))", inputs={"x": before})
    graph, store, root = global_graph(tmp_path / "repair", expression)
    graph.register_context("main", anchor=days[0], information_cutoff=days[-1] + timedelta(days=2))
    old = graph.snapshot()["contexts"]["main"]
    initial = finish(graph.plan_update({"main": {"x": before}}, through=days[-1]))
    kwargs = dict(mode="cutoff_tail_repair.v1", repair_from=days[-1], original_update=initial["identity"], context_id="main",
        original_context=old, replacement_context={**old, "information_cutoff": str(days[-1])},
        original_inputs={"x": before}, replacement_inputs={"x": after}, original_projection=projection,
        replacement_projection=projection, report_id="reentry", report_digest="b" * 64)
    plan, _ = admitted(graph, store, root, initial, after, kwargs)
    result = finish(plan)
    assert max(value["kernel_sessions"] for value in plan.resource_usage["main"].values()) == 400
    assert all(value["computed_output_sessions"] == 1 for value in plan.resource_usage["main"].values())
    fresh, fresh_store, fresh_root = global_graph(tmp_path / "reference", Graph.from_logical_spec(graph.snapshot()["spec"], inputs={"x": after}))
    fresh.register_context("main", **kwargs["replacement_context"])
    reference = finish(fresh.plan_update({"main": {"x": after}}, through=days[-1]))
    assert_frame_equal(store.read_values(result["results"]["main"][root], include_traces=True),
        fresh_store.read_values(reference["results"]["main"][fresh_root], include_traces=True), check_exact=True)


def test_repair_never_raises_loader_limits_or_admits_full_source_loading(tmp_path):
    _, after, graph, store, root, update, kwargs = repair_case(tmp_path)
    plan, proof = admitted(graph, store, root, update, after, kwargs)
    actual = plan._input_window_loader
    limits = ResourceLimits(memory_target_mib=32, cache_mib=0, parallel_nodes=1, batch_rows=100)
    def bounded(*args):
        assert active_resource_limits() == limits
        return actual(*args)
    plan._input_window_loader = bounded
    finish(plan, limits=limits)
    other = graph.plan_update({"main": {"x": after}}, through=after.domain.times.max(),
        input_loader=lambda *_: pytest.fail("full input was loaded"))
    with pytest.raises(ValueError, match="bounded source loading"):
        store.apply_input_proof_rebind(proof, update_plan=other)


def test_derived_region_prefix_admission_is_rechecked_transitively(tmp_path):
    # A negative change flips the final rank, retaining the
    # clipped prefix as a real value_source of the corrected region block.
    _, after, graph, store, root, update, kwargs = repair_case(tmp_path, count=40)
    frame = after.collect(include_traces=True).with_columns(pl.when(pl.col("time") == after.domain.times.max())
        .then(pl.when(pl.col("asset_id") == "a").then(-31.).otherwise(pl.col("value"))).otherwise(pl.col("value")).alias("value"))
    after = Node.from_domain(frame, after.domain, trace_columns=after.trace_columns)
    kwargs["replacement_inputs"] = {"x": after}
    plan, _ = admitted(graph, store, root, update, after, kwargs)
    result = finish(plan)
    import json
    with store.transaction() as connection:
        derived = [identity for identity, body in connection.execute("SELECT identity,plan_json FROM input_proof_rebinding_plans")
            if "region_prefix" in json.loads(body)]
        assert derived
        connection.execute("DELETE FROM input_proof_rebinding_plans WHERE identity=?", (derived[0],))
    for operation in (store.describe, store.read, store.verify_identity):
        with pytest.raises(ValueError, match="admission"):
            operation(result["results"]["main"][root])


@pytest.mark.parametrize("damage", ["header", "unmatched"])
def test_missing_earlier_region_candidate_never_replays_history(tmp_path, monkeypatch, damage):
    _, after, graph, store, root, update, kwargs = repair_case(tmp_path, count=70)
    plan, proof = admitted(graph, store, root, update, after, kwargs)
    while root not in plan.ready("main"):
        for node in plan.ready("main"):
            plan.execute("main", node)
    def forbidden(*_args, **_kwargs):
        pytest.fail("missing retained region evidence triggered a kernel")
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    if damage == "header":
        original_block = proof["nodes"][root]["original_seal"]["blocks"][0]
        with store.transaction() as connection:
            connection.execute("DELETE FROM materialization_headers WHERE identity=?", (original_block,))
    else:
        import bagelquant_core.regions as regions
        match = regions._matching_certificate
        def missing(plan, context, region, certificate, start, end, direct, **kwargs):
            return False if start == after.domain.times.min() else match(plan, context, region, certificate, start, end, direct, **kwargs)
        monkeypatch.setattr(regions, "_matching_certificate", missing)
    with pytest.raises(ValueError, match="retained region prefix evidence"):
        plan.execute("main", root)


@pytest.mark.parametrize("limit", [239, 240, 241])
def test_fill_support_includes_exact_limit_and_excludes_older_null_runs(tmp_path, limit):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(301)]
    frame = pl.DataFrame([(day, str(seed), 1. if i == seed else None, day)
        for i, day in enumerate(days) for seed in (59, 60, 61, 62)],
        schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64, "available_date": pl.Date}, orient="row")
    domain = Domain(calendar=days, universe=["59", "60", "61", "62"])
    before = Node.from_domain(frame, domain, source_key="x", trace_columns=("available_date",))
    after = Node.from_domain(frame.with_columns(pl.when((pl.col("asset_id") == "62") & (pl.col("time") == days[-1]))
        .then(3.).otherwise(pl.col("value")).alias("value")), domain, trace_columns=("available_date",))
    graph, store, root = global_graph(tmp_path / "repair", Graph.from_dsl(f"output = ffill(x, limit={limit})", inputs={"x": before}))
    graph.register_context("main", anchor=days[0], information_cutoff=days[-1] + timedelta(days=2))
    old = graph.snapshot()["contexts"]["main"]
    update = finish(graph.plan_update({"main": {"x": before}}, through=days[-1]))
    kwargs = dict(mode="cutoff_tail_repair.v1", repair_from=days[-1], original_update=update["identity"], context_id="main",
        original_context=old, replacement_context={**old, "information_cutoff": str(days[-1])},
        original_inputs={"x": before}, replacement_inputs={"x": after}, original_projection=projection,
        replacement_projection=projection, report_id="fill-boundary", report_digest="c" * 64)
    plan, _ = admitted(graph, store, root, update, after, kwargs)
    result = finish(plan)
    reference = Graph.from_logical_spec(graph.snapshot()["spec"], inputs={"x": after}).compute()
    assert_frame_equal(store.read_values(result["results"]["main"][root], include_traces=True),
        reference.collect(dense=False, include_traces=True), check_exact=True)
    final = store.read_values(result["results"]["main"][root], start=days[-1])
    assert dict(final.select("asset_id", "value").iter_rows()) == {
        "59": 1. if limit >= 241 else None, "60": 1. if limit >= 240 else None, "61": 1., "62": 3.}


def test_region_prefix_survives_cancellation_without_historical_replay(tmp_path, monkeypatch):
    _, after, graph, store, root, update, kwargs = repair_case(tmp_path, count=40)
    plan, _ = admitted(graph, store, root, update, after, kwargs)
    while root not in plan.ready("main"):
        for node in plan.ready("main"):
            plan.execute("main", node)
    import bagelquant_core.regions as regions
    original = regions._run_finite_outputs
    def canceled(*_args, **_kwargs):
        raise RuntimeError("canceled after prefix")
    monkeypatch.setattr(regions, "_run_finite_outputs", canceled)
    with pytest.raises(RuntimeError, match="canceled after prefix"):
        plan.execute("main", root)
    import json
    with store.read_transaction() as connection:
        assert any("region_prefix" in json.loads(row[0]) for row in connection.execute("SELECT plan_json FROM input_proof_rebinding_plans"))
    def tail_only(spec, nodes, scoped, parameters, start, guard):
        assert start == after.domain.times.max()
        return original(spec, nodes, scoped, parameters, start, guard)
    monkeypatch.setattr(regions, "_run_finite_outputs", tail_only)
    retry, _ = admitted(graph, store, root, update, after, kwargs)
    finish(retry)
    assert all(value["computed_output_sessions"] == 1 for value in retry.resource_usage["main"].values())
