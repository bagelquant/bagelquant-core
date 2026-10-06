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
def test_training_checkpoint_reuse_requires_all_parent_evidence(tmp_path, scenario):
    root = tmp_path
    first = sources(18)
    g, store, node = create(root / "inc", first)
    firstplan, old = finish(g, first)
    current = sources(26, None if scenario == "append" else scenario)
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
    assert (use["checkpoint_through"] is not None) == (scenario == "append"), (
        scenario,
        use,
    )
