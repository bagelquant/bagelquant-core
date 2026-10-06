from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, Graph, Node, equal_weight, rebalance, rebalance_value, top_n
from bagelquant_core.optimization import solve_regularized_weights
import numpy as np


def _inputs():
    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(12)]
    frame = pl.DataFrame([{"time": day, "asset_id": asset, "value": 1.0}
                         for day in days for asset in ("A", "B", "C")])
    domain = Domain(calendar=days, universe=["A", "B", "C"])
    return days, frame, Node.from_domain(frame, domain, name="prediction")


def test_top_n_ties_full_exit_and_insufficient_cross_section():
    days, frame, panel = _inputs()
    graph = equal_weight(top_n(panel, count=2))
    result = graph.compute().collect(dense=True)
    assert result.filter(pl.col("time") == days[0])["value"].to_list() == [0.5, 0.5, 0.0]
    missing = frame.with_columns(pl.when(pl.col("asset_id").is_in(["B", "C"]))
                                .then(None).otherwise(pl.col("value")).alias("value"))
    unavailable = equal_weight.operation(top_n.operation(missing, count=2))
    value = rebalance_value(unavailable, calendar=frame, data_start=days[0], every=5)
    assert value.weights.is_empty()
    assert value.decisions["status"].to_list() == ["unavailable", "hold", "hold", "hold", "hold", "unavailable", "hold", "hold", "hold", "hold", "unavailable", "hold"]
    assert value.decisions.filter(pl.col("status") == "unavailable")["reason"].null_count() == 0


def test_rebalance_calendar_is_explicit_and_slice_does_not_move_anchor():
    days, frame, panel = _inputs()
    calendar = Node.from_domain(frame, panel.domain, name="calendar")
    graph = rebalance(equal_weight(top_n(panel, count=2)), every=5,
                      calendar=calendar, data_start=days[0].isoformat())
    specification = graph.spec().to_dict()
    assert specification["nodes"][-1]["panel_parameters"] == {"calendar": ["calendar"]}
    assert 'calendar' in graph.mermaid()
    restored = Graph.compile(specification).compute({"prediction": panel, "calendar": calendar}, dense_output=False)
    targets = restored.collect(dense=False).filter(pl.col("value").is_not_null())
    assert targets["time"].unique().sort().to_list() == [days[0], days[5], days[10]]
    weights = equal_weight.operation(top_n.operation(frame, count=2))
    whole = rebalance_value(weights, calendar=frame, data_start=days[0], every=5)
    suffix = rebalance_value(weights.filter(pl.col("time") >= days[3]), calendar=frame, data_start=days[0], every=5)
    assert_frame_equal(suffix.weights, whole.weights.filter(pl.col("time") >= days[3]))
    assert_frame_equal(suffix.decisions, whole.decisions.filter(pl.col("time") >= days[3]))


def test_optimizer_rejects_infeasible_capacity():
    with pytest.raises(ValueError, match="capacity"):
        solve_regularized_weights(np.array([1., 2.]), np.zeros(2),
            concentration_penalty=10, turnover_penalty=0.1, max_weight=0.4, tolerance=1e-7)


def test_empty_universe_decisions_keep_the_full_anchored_calendar():
    days,frame,_=_inputs()
    value=rebalance_value(frame.head(0),calendar=frame,data_start=days[0],every=5,
        coverage_calendar=pl.DataFrame({'time':days}))
    assert value.decisions.height==len(days)
    assert value.decisions.filter(pl.col('status')=='unavailable')['time'].to_list()==[days[0],days[5],days[10]]
    assert value.decisions.filter(pl.col('status')=='unavailable')['reason'].unique().to_list()==['empty_universe']
