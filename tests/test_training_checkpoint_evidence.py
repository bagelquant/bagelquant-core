from datetime import date, timedelta
import pytest
import polars as pl
from polars.testing import assert_frame_equal
from bagelquant_core import Node, Domain, Graph, CoreStore
from bagelquant_core.operator import rolling_elastic_net_prediction


def sources(n, change=None):
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    rows = [
        {
            "time": day,
            "asset_id": a,
            "value": float((i * 3 + j) % 17) + j * 0.13,
            "available": day,
        }
        for i, day in enumerate(dates)
        for j, a in enumerate("ABCD")
    ]
    frame = pl.DataFrame(rows)
    universe = frame.select("time", "asset_id")
    if change == "membership":
        universe = universe.filter(
            ~((pl.col("time") == dates[5]) & (pl.col("asset_id") == "B"))
        )
    domain = Domain(
        calendar=dates, universe=universe.with_columns(pl.lit(True).alias("active"))
    )
    labels = frame.with_columns(
        (
            pl.col("value") * 0.7
            + pl.col("asset_id").replace_strict(
                {"A": 0.2, "B": -0.2, "C": 0.1, "D": -0.1}
            )
        ).alias("value")
    )
    if change == "revision":
        frame = frame.with_columns(
            pl.when(pl.col("time") == dates[5])
            .then(pl.col("value") + 2)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    if change == "trace":
        frame = frame.with_columns(
            pl.when(pl.col("time") == dates[5])
            .then(dates[9])
            .otherwise(pl.col("available"))
            .alias("available")
        )
    if change == "label_revision":
        labels = labels.with_columns(
            pl.when(pl.col("time") == dates[5])
            .then(pl.col("value") * 2)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    if change == "matured_labels":
        labels = labels.with_columns(
            pl.when(pl.col("time") >= dates[16])
            .then(pl.col("value") + 10)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    end = frame.with_columns(
        (pl.col("time").dt.epoch("d") + date(1970, 1, 1).toordinal() + 2)
        .cast(pl.Float64)
        .alias("value")
    )
    available = end
    if change == "label_available":
        available = end.with_columns(
            pl.when(pl.col("time") == dates[5])
            .then(float(dates[15].toordinal()))
            .otherwise(pl.col("value"))
            .alias("value")
        )
    return {
        key: Node.from_domain(f, domain, source_key=key, trace_columns=("available",))
        for key, f in {
            "x": frame,
            "labels": labels,
            "label_end": end,
            "label_available": available,
        }.items()
    }


def expression(s):
    return rolling_elastic_net_prediction(
        s["x"],
        labels=s["labels"],
        label_end=s["label_end"],
        label_available=s["label_available"],
        window=8,
        fit_every=3,
        min_samples=4,
        max_samples=20,
        label_maturity=2,
        alpha=0.01,
    )


def create(root, s):
    store = CoreStore(root / "m.sqlite", root / "a")
    graph = Graph(store=store, graph_id="test")
    node = next(iter(graph.merge(expression(s)).outputs.values()))
    graph.register_context("main", anchor="2024-01-01")
    return graph, store, node


def finish(graph, s):
    plan = graph.plan_update({"main": s}, through=s["x"].domain.times.max())
    while not plan.complete:
        for node in plan.ready("main"):
            plan.execute("main", node)
    return plan, plan.publish()


def test_new_unconsumed_labels_preserve_old_fits_with_complete_feature_proof(tmp_path):
    from bagelquant_core.operator.training import capture_training_audits
    def inputs(count, change=None):
        values = sources(count, change)
        for key in ("labels", "label_end", "label_available"):
            node = values[key]
            values[key] = Node.from_domain(node.collect().select("time", "asset_id", "value"),
                node.domain, source_key=key)
        return values
    first = inputs(18)
    graph, store, root = create(tmp_path / "incremental", first)
    finish(graph, first)
    current = inputs(26, "matured_labels")
    with capture_training_audits() as new_fits:
        plan, updated = finish(graph, current)
    assert plan.resource_usage["main"][root]["checkpoint_through"] == str(first["x"].domain.times.max())
    assert new_fits and all(audit["fit_date"] > first["x"].domain.times.max() for audit in new_fits)
    reference, fresh, reference_root = create(tmp_path / "reference", current)
    _, complete = finish(reference, current)
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True),
        fresh.read_values(complete["results"]["main"][reference_root], include_traces=True), check_exact=True)
    assert store.evidence(updated["results"]["main"][root])["training_audits"] == fresh.evidence(
        complete["results"]["main"][reference_root])["training_audits"]


def test_training_feature_aliased_as_label_keeps_complete_prediction_prefix(tmp_path):
    def input(count, revise=False):
        source = sources(count)["x"]
        frame = source.collect().select("time", "asset_id", "value")
        if revise:
            frame = frame.with_columns(pl.when(pl.col("time") == source.domain.times[16])
                .then(pl.col("value") + 5).otherwise(pl.col("value")).alias("value"))
        return Node.from_domain(frame, source.domain, source_key="x")
    first = input(18)
    store = CoreStore(tmp_path / "meta.sqlite", tmp_path / "values")
    graph = Graph(store=store, graph_id="alias")
    graph.merge(rolling_elastic_net_prediction(first, labels=first, window=8, fit_every=3,
        min_samples=4, max_samples=20, label_maturity=2, alpha=0.01))
    graph.register_context("main", anchor="2024-01-01")
    finish(graph, {"x": first})
    with pytest.raises(ValueError, match="proven preceding state"):
        finish(graph, {"x": input(26, revise=True)})


def test_consumption_cutoff_proof_preserves_fits_when_future_label_endpoints_extend(tmp_path):
    from bagelquant_core.hashing import hash_dataframe
    from bagelquant_core.operator.training import capture_training_audits
    def inputs(count):
        result = sources(count)
        last = result["x"].domain.times.max().toordinal()
        ends = result["label_end"].collect().select("time", "asset_id",
            (pl.col("value") + 3).alias("value"))
        visible = ends.with_columns(pl.when(pl.col("value") <= last).then(pl.col("value")).otherwise(None).alias("value"))
        labels = result["labels"].collect().select("time", "asset_id", "value").join(
            visible.rename({"value": "end"}), on=["time", "asset_id"]).with_columns(
            pl.when(pl.col("end").is_not_null()).then(pl.col("value")).otherwise(None).alias("value")).drop("end")
        for key, frame in {"labels": labels, "label_end": visible, "label_available": visible}.items():
            result[key] = Node.from_domain(frame, result[key].domain, source_key=key)
        return result
    def finish_scoped(graph, values):
        def scoped(_context, key, start, end, cutoff):
            if key == "x":
                return values[key].interval_identity(start, end)
            frame = values[key].collect().filter(pl.col("time").is_between(start, end))
            maturity = values["label_end"].collect().select("time", "asset_id", pl.col("value").alias("end"))
            return hash_dataframe(frame.join(maturity, on=["time", "asset_id"]).filter(
                pl.col("end") <= cutoff.toordinal()).drop("end"))
        plan = graph.plan_update({"main": values}, through=values["x"].domain.times.max(),
            input_proof=lambda _context, key, start, end: values[key].interval_identity(start, end),
            input_scope_proof=scoped)
        while not plan.complete:
            for node in plan.ready("main"):
                plan.execute("main", node)
        return plan, plan.publish()
    before, after = inputs(18), inputs(26)
    assert before["labels"].interval_identity(date(2024, 1, 1), date(2024, 1, 14)) != after["labels"].interval_identity(
        date(2024, 1, 1), date(2024, 1, 14))
    graph, store, root = create(tmp_path / "incremental", before)
    finish_scoped(graph, before)
    with capture_training_audits() as fits:
        plan, updated = finish_scoped(graph, after)
    assert plan.resource_usage["main"][root]["checkpoint_through"] == "2024-01-18"
    assert fits and all(row["fit_date"] > date(2024, 1, 18) for row in fits)
    reference, fresh, reference_root = create(tmp_path / "reference", after)
    _, full = finish_scoped(reference, after)
    assert_frame_equal(store.read_values(updated["results"]["main"][root], include_traces=True),
        fresh.read_values(full["results"]["main"][reference_root], include_traces=True), check_exact=True)
    assert store.evidence(updated["results"]["main"][root])["training_audits"] == fresh.evidence(
        full["results"]["main"][reference_root])["training_audits"]


@pytest.mark.parametrize(
    "scenario",
    (
        "append",
        "revision",
        "membership",
        "trace",
        "label_revision",
        "label_available",
        "matured_labels",
    ),
)
@pytest.mark.parametrize("preceding_checkpoint", [False, True])
def test_training_checkpoint_reuse_requires_all_parent_evidence(tmp_path, scenario, preceding_checkpoint):
    root = tmp_path
    first = sources(18)
    g, store, node = create(root / "inc", first)
    if preceding_checkpoint:
        finish(g, sources(3))
    firstplan, old = finish(g, first)
    current = sources(26, None if scenario == "append" else scenario)
    if scenario != "append" and not preceding_checkpoint:
        with pytest.raises(ValueError, match="proven preceding state"):
            finish(g, current)
        assert store.graph_state("test")["current_update"] == old["identity"]
        return
    plan, receipt = finish(g, current)
    actual = store.read(receipt["results"]["main"][node])
    fresh, fs, fn = create(root / "fresh", current)
    _, full = finish(fresh, current)
    expected = fs.read(full["results"]["main"][fn])
    assert_frame_equal(
        actual.panel.collect(include_traces=True),
        expected.panel.collect(include_traces=True),
        check_exact=True,
    )
    assert actual.checkpoint == expected.checkpoint, (scenario, "checkpoint")
    assert actual.training_audits == expected.training_audits, (scenario, "audits")
    assert all(
        audit.get("maximum_label_available", audit["fit_date"]) <= audit["fit_date"]
        for audit in actual.training_audits
        if audit.get("maximum_label_available")
    )
    use = plan.resource_usage["main"][node]
    assert use["checkpoint_through"] == ("2024-01-18" if scenario == "append" else "2024-01-03")
