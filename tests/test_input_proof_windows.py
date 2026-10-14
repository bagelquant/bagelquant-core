"""Metadata proof-scope plans share actual execution interval contracts."""

from datetime import date, timedelta

import polars as pl
import pytest

from bagelquant_core import Domain, Node, ExecutionRuntime
from bagelquant_core.operator import add, ewm_mean, project, rolling_mean
from test_global_graph import finish, global_graph, source
from test_training_checkpoint_evidence import sources, expression


def tuples(frame):
    return {
        (row["input_key"], row["start"], row["end"], row["consumption_cutoff"])
        for row in frame.to_dicts()
        if row["status"] not in ("unknown", "empty")
    }


def sparse(count):
    days = [date(2024, 1, 1) + timedelta(days=2 * i) for i in range(count)]
    frame = pl.DataFrame(
        {
            "time": [day for i, day in enumerate(days) for _ in range(2)],
            "asset_id": ["a", "b"] * count,
            "value": [float(i) for i in range(2 * count)],
        }
    )
    frame = frame.filter(
        (pl.col("asset_id") == "a")
        | (pl.col("time") <= days[9])
        | (pl.col("time") >= days[min(38, count - 1)])
    )
    domain = Domain(
        calendar=days,
        universe=frame.select("time", "asset_id").with_columns(
            pl.lit(True).alias("active")
        ),
    )
    return Node.from_domain(frame, domain, source_key="x")


@pytest.mark.parametrize("dynamic", [False, True])
def test_finite_windows_equal_actual_callback_set_with_retained_partial(
    tmp_path, dynamic
):
    make = sparse if dynamic else lambda n: source(n)
    before = make(35)
    graph, store, root = global_graph(tmp_path, rolling_mean(before, window=3))
    calls = []

    def plan(value):
        def proof(_context, key, start, end):
            calls.append((key, start, end, None))
            return value.interval_identity(start, end)

        return graph.plan_update(
            {"main": {"x": value}}, through=value.domain.times.max(), input_proof=proof
        )

    first = plan(before)
    expected = tuples(first.input_proof_windows("main"))
    assert not calls
    finish(first)
    assert set(calls) == expected
    calls.clear()
    current = make(40)
    next_plan = plan(current)
    windows = next_plan.input_proof_windows("main")
    assert not calls
    assert "candidate" in windows["status"].to_list()
    expected = tuples(windows)
    finish(next_plan)
    assert set(calls) == expected
    if dynamic:
        # Asset b's long gap extends the actual observation-count halo.
        second = windows.filter(pl.col("end") == current.domain.times.max())
        assert second["start"][0] < current.domain.times[30]
    assert next_plan.input_proof_windows("main").is_empty()
    assert store.graph_state("global")["current_update"] is not None


def test_named_auxiliary_external_input_is_in_finite_scope_plan(tmp_path):
    like = source(35, name="like")
    index = Node.from_domain(
        like.collect(include_traces=True).filter(pl.col("asset_id") == "a"),
        like.domain,
        source_key="index",
        trace_columns=("available_date",),
    )
    graph, _, _ = global_graph(tmp_path, project(index, binary=like))
    calls = []
    values = {"index": index, "like": like}

    def proof(_context, key, start, end):
        calls.append((key, start, end, None))
        return values[key].interval_identity(start, end)

    plan = graph.plan_update(
        {"main": values}, through=like.domain.times.max(), input_proof=proof
    )
    windows = plan.input_proof_windows("main")
    assert set(windows["input_key"]) == {"index", "like"}
    expected = tuples(windows)
    finish(plan)
    assert set(calls) == expected


def test_different_parent_calendar_does_not_advertise_finite_contract(tmp_path):
    x = source(35)
    frame = x.collect(include_traces=True).filter(pl.col("time") != date(2024, 1, 10))
    y = Node.from_domain(
        frame,
        Domain(
            calendar=x.domain.times.filter(x.domain.times != date(2024, 1, 10)),
            universe=["a", "b"],
            identity=x.domain.signature,
        ),
        source_key="y",
        trace_columns=("available_date",),
    )
    graph, _, root = global_graph(tmp_path, add(x, y))
    plan = graph.plan_update({"main": {"x": x, "y": y}}, through=x.domain.times.max())
    assert root not in plan.policies["main"]
    windows = plan.input_proof_windows("main")
    assert set(windows["status"]) == {"unknown"}
    assert windows["start"].null_count() == 2 and windows["end"].null_count() == 2


def test_checkpoint_current_and_saved_prefix_equal_actual_callback_set(tmp_path):
    before = source(18)
    graph, _, _ = global_graph(tmp_path, ewm_mean(before, alpha=0.5))
    calls = []

    def plan(value):
        def proof(_context, key, start, end):
            calls.append((key, start, end, None))
            return value.interval_identity(start, end)

        return graph.plan_update(
            {"main": {"x": value}}, through=value.domain.times.max(), input_proof=proof
        )

    finish(plan(before))
    calls.clear()
    current = source(26)
    next_plan = plan(current)
    expected = tuples(next_plan.input_proof_windows("main"))
    assert expected == {
        ("x", date(2024, 1, 1), date(2024, 1, 18), None),
        ("x", date(2024, 1, 1), date(2024, 1, 26), None),
    }
    finish(next_plan)
    assert set(calls) == expected


@pytest.mark.parametrize("trace_aux", [False, True])
def test_training_scopes_preserve_actual_fit_cutoff_and_trace_guard(
    tmp_path, trace_aux
):
    values = sources(18)
    if not trace_aux:
        values = {
            key: (
                value
                if key == "x"
                else Node.from_domain(value.collect(), value.domain, source_key=key)
            )
            for key, value in values.items()
        }
    graph, _, root = global_graph(tmp_path, expression(values))
    calls = []

    def proof(_context, key, start, end):
        calls.append((key, start, end, None))
        return values[key].interval_identity(start, end)

    def scoped(_context, key, start, end, cutoff):
        calls.append((key, start, end, cutoff))
        return values[key].interval_identity(start, end)

    plan = graph.plan_update(
        {"main": values},
        through=date(2024, 1, 18),
        input_proof=proof,
        input_scope_proof=scoped,
    )
    windows = plan.input_proof_windows("main")
    assert not calls
    label = windows.filter(pl.col("input_key") == "labels").row(0, named=True)
    if trace_aux:
        assert label["end"] == date(2024, 1, 18) and label["consumption_cutoff"] is None
    else:
        assert label["end"] == date(2024, 1, 14) and label[
            "consumption_cutoff"
        ] == date(2024, 1, 16)
    expected = tuples(windows)
    finish(plan)
    assert set(calls) == expected
    assert root in plan.results["main"]


def test_scope_api_forbids_token_math_load_audit_or_write(tmp_path, monkeypatch):
    x = source(35)
    graph, store, _ = global_graph(tmp_path, rolling_mean(x, window=3))

    def forbidden(*_a, **_k):
        pytest.fail("metadata scope planning performed numerical or mutation work")

    plan = graph.plan_update(
        {"main": {"x": x}},
        through=x.domain.times.max(),
        input_proof=forbidden,
        input_loader=forbidden,
    )
    monkeypatch.setattr(ExecutionRuntime, "run", forbidden)
    monkeypatch.setattr(store, "lookup", forbidden)
    monkeypatch.setattr(store, "verify", forbidden)
    monkeypatch.setattr(store, "publish", forbidden)
    monkeypatch.setattr(store, "read", forbidden)
    before = store.graph_state("global")
    windows = plan.input_proof_windows("main")
    assert windows.schema["start"] == pl.Date and windows.height == 2
    assert store.graph_state("global") == before
    assert not store.inventory()
    with pytest.raises(KeyError, match="unknown execution context"):
        plan.input_proof_windows("other")


def test_training_consumes_scoped_only_callback_without_unscoped_override(tmp_path):
    values = {key: (value if key == "x" else Node.from_domain(value.collect(), value.domain, source_key=key))
        for key, value in sources(18).items()}
    graph, _, root = global_graph(tmp_path, expression(values))
    calls = []
    def scoped(_context, key, start, end, cutoff):
        calls.append((key, start, end, cutoff))
        return values[key].interval_identity(start, end)
    plan = graph.plan_update({"main": values}, through=date(2024, 1, 18), input_scope_proof=scoped)
    expected = {row for row in tuples(plan.input_proof_windows("main")) if row[-1] is not None}
    finish(plan)
    assert expected and set(calls) == expected
    assert root in plan.results["main"]


def test_explicit_consumption_cutoff_cannot_borrow_an_unscoped_proof(tmp_path):
    value = source(18)
    graph, _, _ = global_graph(tmp_path, rolling_mean(value, window=3))
    def unscoped(*args):
        pytest.fail("a causal cutoff borrowed unscoped evidence")
    plan = graph.plan_update({"main": {"x": value}}, through=value.domain.times.max(), input_proof=unscoped)
    parent = next(node.node_id for node in plan.spec.nodes if node.node_type == "input")
    with pytest.raises(ValueError, match="explicit scoped input proof"):
        plan._proof_reader("main")(parent, value, date(2024, 1, 1), date(2024, 1, 14), date(2024, 1, 16))


def test_training_empty_and_feature_alias_have_no_invented_label_scope(tmp_path):
    tiny = sources(2)
    graph, _, _ = global_graph(tmp_path / "empty", expression(tiny))
    # Remove auxiliary traces to activate the empty consumed-label contract.
    tiny = {
        key: (
            value
            if key == "x"
            else Node.from_domain(value.collect(), value.domain, source_key=key)
        )
        for key, value in tiny.items()
    }
    # Rebind the same frozen logical graph to these traceless metadata sources.
    plan = graph.plan_update(
        {"main": tiny}, through=date(2024, 1, 2), input_scope_proof=lambda *_: None
    )
    windows = plan.input_proof_windows("main")
    labels = windows.filter(pl.col("input_key") != "x")
    assert set(labels["status"]) == {"empty"} and labels["end"].null_count() == 3
    from bagelquant_core.operator import rolling_elastic_net_prediction

    feature = source(18)
    aliased = rolling_elastic_net_prediction(
        feature,
        labels=feature,
        window=8,
        fit_every=3,
        min_samples=4,
        max_samples=20,
        label_maturity=2,
        alpha=0.01,
    )
    graph, _, _ = global_graph(tmp_path / "aliased", aliased)
    plan = graph.plan_update(
        {"main": {"x": feature}},
        through=feature.domain.times.max(),
        input_scope_proof=lambda *_: None,
    )
    windows = plan.input_proof_windows("main")
    assert windows.height == 1 and windows["end"][0] == date(2024, 1, 18)
    assert windows["consumption_cutoff"][0] is None


def test_checkpoint_retained_scope_preserves_legacy_absent_fields(tmp_path):
    from dataclasses import replace
    from bagelquant_core import NodeMaterialization

    values = sources(18)
    values = {
        key: (
            value
            if key == "x"
            else Node.from_domain(value.collect(), value.domain, source_key=key)
        )
        for key, value in values.items()
    }
    graph, store, root = global_graph(tmp_path, expression(values))
    prior = finish(graph.plan_update({"main": values}, through=date(2024, 1, 18)))
    identity = prior["results"]["main"][root]
    record = store.read(identity)
    original_policy = store.manifest(identity)["execution_policy"]
    legacy = {
        key: value
        for key, value in original_policy.items()
        if key not in ("parent_ends", "parent_cutoffs")
    }
    store.publish(
        NodeMaterialization(
            replace(record.key, context_identity="legacy-full-parent-scope"),
            record.panel,
            checkpoint=record.checkpoint,
            training_audits=record.training_audits,
        ),
        execution_policy=legacy,
    )
    current = sources(26)
    current = {
        key: (
            value
            if key == "x"
            else Node.from_domain(value.collect(), value.domain, source_key=key)
        )
        for key, value in current.items()
    }
    plan = graph.plan_update(
        {"main": current}, through=date(2024, 1, 26), input_scope_proof=lambda *_: None
    )
    windows = plan.input_proof_windows("main")
    retained = windows.filter(
        (pl.col("input_key") == "labels") & (pl.col("status") == "candidate")
    )
    # The native old consumed endpoint and the legacy full endpoint are both
    # possible prerequisites; the latter must not be narrowed retroactively.
    assert set(retained["end"]) == {date(2024, 1, 14), date(2024, 1, 18)}
    assert (
        retained.filter(pl.col("end") == date(2024, 1, 18))["consumption_cutoff"][0]
        is None
    )


def test_checkpoint_irrelevant_parameters_deduplicate_retained_scope(tmp_path):
    from dataclasses import replace
    from bagelquant_core import NodeMaterialization

    first = source(18)
    graph, store, root = global_graph(tmp_path, ewm_mean(first, alpha=0.5))
    prior = finish(
        graph.plan_update({"main": {"x": first}}, through=first.domain.times.max())
    )
    record = store.read(prior["results"]["main"][root])
    policy = store.manifest(record.key.identity)["execution_policy"]
    fake = {**policy, "parameters": {"$other": 3}}
    store.publish(
        NodeMaterialization(
            replace(record.key, context_identity="foreign-bindings"),
            record.panel,
            checkpoint=record.checkpoint,
        ),
        execution_policy=fake,
    )
    # Literal operators use no $ bindings, so irrelevant parameters are safe
    # and do not create a second scope: both receipts describe the same prefix.
    current = source(26)
    plan = graph.plan_update(
        {"main": {"x": current}}, through=current.domain.times.max()
    )
    assert plan.input_proof_windows("main").height == 2


def test_checkpoint_changed_relevant_binding_excludes_old_candidate(tmp_path):
    first = source(18)
    graph, _, _ = global_graph(tmp_path, ewm_mean(first, alpha="$alpha"))
    graph.register_context("main", anchor="2024-01-01", parameters={"$alpha": 0.5})
    finish(graph.plan_update({"main": {"x": first}}, through=first.domain.times.max()))
    graph.register_context("main", anchor="2024-01-01", parameters={"$alpha": 0.3})
    current = source(26)
    plan = graph.plan_update(
        {"main": {"x": current}}, through=current.domain.times.max()
    )
    windows = plan.input_proof_windows("main")
    assert windows.height == 1 and windows["status"][0] == "required"
    assert windows["end"][0] == date(2024, 1, 26)
