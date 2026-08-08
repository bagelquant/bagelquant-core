from __future__ import annotations

import importlib
from datetime import date

import numpy as np
import polars as pl
import pytest

from bagelquant_core import (
    Domain,
    EqualWeightPredictionComposer,
    GLSPredictionComposer,
    Graph,
    ICWeightedPredictionComposer,
    IdentityPredictionComposer,
    OLSPredictionComposer,
    Panel,
    PredictionPanel,
    PredictionTrainingContext,
)
from bagelquant_core.transformer import rank


def _panel(
    domain: Domain,
    name: str,
    values: list[tuple[date, str, float | None]],
) -> Panel:
    return Panel.from_domain(
        pl.DataFrame(
            values,
            schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64},
            orient="row",
        ),
        domain,
        name=name,
    )


def test_identity_prediction_is_typed_terminal_and_round_trips_spec() -> None:
    day = date(2024, 1, 31)
    domain = Domain(calendar=[day], universe=["A", "B"])
    alpha = _panel(domain, "alpha", [(day, "A", 1.0), (day, "B", 3.0)])

    graph = IdentityPredictionComposer().compose(alpha)
    result = graph.compute(dense_output=False)

    assert isinstance(result, PredictionPanel)
    assert result.collect(dense=False).get_column("value").to_list() == pytest.approx(
        [1.0, 3.0]
    )
    compiled = Graph.compile(graph.spec().to_dict())
    assert isinstance(compiled.compute({"alpha": alpha}, dense_output=False), PredictionPanel)
    with pytest.raises(ValueError, match="terminal"):
        rank(graph)


def test_equal_weight_renormalizes_available_alpha_values() -> None:
    day = date(2024, 1, 31)
    domain = Domain(calendar=[day], universe=["A", "B", "C"])
    first = _panel(
        domain,
        "first",
        [(day, "A", 1.0), (day, "B", 2.0), (day, "C", 3.0)],
    )
    second = _panel(
        domain,
        "second",
        [(day, "A", 2.0), (day, "B", None), (day, "C", 4.0)],
    )

    result = EqualWeightPredictionComposer().compose(first, second).compute(
        dense_output=False
    )

    assert result.collect(dense=False).get_column("value").to_list() == pytest.approx(
        [1.5, 2.0, 3.5]
    )


def test_prediction_composer_rejects_removed_standardization_parameter() -> None:
    day = date(2024, 1, 31)
    domain = Domain(calendar=[day], universe=["A", "B", "C", "D"])
    alpha = _panel(
        domain,
        "alpha",
        [(day, "A", 1.0), (day, "B", 1.0), (day, "C", 3.0), (day, "D", None)],
    )

    with pytest.raises(TypeError, match="standardization"):
        IdentityPredictionComposer().compose(alpha, standardization="percentile_rank")


def test_prediction_composition_excludes_non_finite_alpha_values() -> None:
    day = date(2024, 1, 31)
    domain = Domain(calendar=[day], universe=["A", "B", "C"])
    alpha = _panel(
        domain,
        "alpha",
        [(day, "A", 1.0), (day, "B", 2.0), (day, "C", float("inf"))],
    )

    result = IdentityPredictionComposer().compose(alpha).compute(
        dense_output=False
    ).collect(dense=False)

    assert result.get_column("asset_id").to_list() == ["A", "B"]
    assert result.get_column("value").to_list() == pytest.approx([1.0, 2.0])


def test_removed_signal_module_has_no_compatibility_alias() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("bagelquant_core.signal")


def test_ic_weighted_uses_only_positive_full_window_ic() -> None:
    times = [date(2024, month, 1) for month in range(1, 4)]
    assets = ["A", "B", "C"]
    domain = Domain(calendar=times, universe=assets)
    positive = [
        (current, asset, float(index + 1))
        for current in times
        for index, asset in enumerate(assets)
    ]
    negative = [
        (current, asset, float(len(assets) - index))
        for current in times
        for index, asset in enumerate(assets)
    ]
    targets = [
        (current, asset, float(index + 1))
        for current in times[:2]
        for index, asset in enumerate(assets)
    ]
    availability = [
        (current, asset, float(times[time_index + 1].toordinal()))
        for time_index, current in enumerate(times[:2])
        for asset in assets
    ]

    result = (
        ICWeightedPredictionComposer(2)
        .compose(
            _panel(domain, "positive", positive),
            _panel(domain, "negative", negative),
            training=PredictionTrainingContext(
                _panel(domain, "targets", targets),
                _panel(domain, "availability", availability),
            ),
        )
        .compute(dense_output=False)
        .collect(dense=False)
    )

    assert result.get_column("time").unique().to_list() == [times[2]]
    assert result.get_column("value").to_list() == pytest.approx([1.0, 2.0, 3.0])


def test_gls_combines_two_invertible_period_covariances() -> None:
    times = [date(2024, month, 1) for month in range(1, 4)]
    assets = ["A", "B", "C", "D", "E", "F"]
    domain = Domain(calendar=times, universe=assets)
    first_values = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    second_values = [0.0, 1.0, 4.0, 1.0, 0.0, 2.0]
    residuals = (
        [-0.2, 0.1, 0.2, -0.1, 0.1, -0.1],
        [0.1, -0.2, 0.1, 0.2, -0.1, -0.1],
    )
    first_rows = [
        (current, asset, first_values[index])
        for current in times
        for index, asset in enumerate(assets)
    ]
    second_rows = [
        (current, asset, second_values[index])
        for current in times
        for index, asset in enumerate(assets)
    ]
    first_z = np.array(first_values)
    first_z = (first_z - first_z.mean()) / first_z.std(ddof=1)
    second_z = np.array(second_values)
    second_z = (second_z - second_z.mean()) / second_z.std(ddof=1)
    target_rows = []
    availability_rows = []
    for period_index, current in enumerate(times[:2]):
        for asset_index, asset in enumerate(assets):
            target_rows.append(
                (
                    current,
                    asset,
                    float(
                        0.5
                        + 2.0 * first_z[asset_index]
                        - second_z[asset_index]
                        + residuals[period_index][asset_index]
                    ),
                )
            )
            availability_rows.append(
                (current, asset, float(times[period_index + 1].toordinal()))
            )

    result = (
        GLSPredictionComposer(2)
        .compose(
            _panel(domain, "first", first_rows),
            _panel(domain, "second", second_rows),
            training=PredictionTrainingContext(
                _panel(domain, "targets", target_rows),
                _panel(domain, "availability", availability_rows),
            ),
        )
        .compute(dense_output=False)
        .collect(dense=False)
    )

    assert result.get_column("time").unique().to_list() == [times[2]]
    assert result.height == len(assets)
    assert result.get_column("value").is_finite().all()


def test_ols_waits_for_available_full_period_window() -> None:
    times = [date(2024, month, 1) for month in range(1, 5)]
    assets = ["A", "B", "C", "D"]
    domain = Domain(calendar=times, universe=assets)
    alpha_rows: list[tuple[date, str, float]] = []
    alpha_two_rows: list[tuple[date, str, float]] = []
    target_rows: list[tuple[date, str, float]] = []
    availability_rows: list[tuple[date, str, float]] = []
    for time_index, current in enumerate(times):
        for asset_index, asset in enumerate(assets):
            first = float(asset_index + 1)
            second = float((asset_index + 1) ** 2)
            alpha_rows.append((current, asset, first))
            alpha_two_rows.append((current, asset, second))
            if time_index < 3:
                target_rows.append((current, asset, 0.5 + first + 0.25 * second))
                availability_rows.append(
                    (current, asset, float(times[time_index + 1].toordinal()))
                )
    first = _panel(domain, "first", alpha_rows)
    second = _panel(domain, "second", alpha_two_rows)
    targets = _panel(domain, "targets", target_rows)
    availability = _panel(domain, "availability", availability_rows)

    result = OLSPredictionComposer(2).compose(
        first,
        second,
        training=PredictionTrainingContext(targets, availability),
    ).compute(dense_output=False).collect(dense=False)

    assert result.get_column("time").unique().to_list() == [times[2], times[3]]
    assert result.filter(pl.col("time") == times[3]).height == len(assets)


def test_ols_requires_consecutive_completed_training_periods() -> None:
    times = [date(2024, month, 1) for month in range(1, 6)]
    assets = ["A", "B", "C", "D"]
    domain = Domain(calendar=times, universe=assets)
    first_rows: list[tuple[date, str, float]] = []
    second_rows: list[tuple[date, str, float]] = []
    target_rows: list[tuple[date, str, float]] = []
    availability_rows: list[tuple[date, str, float]] = []
    for time_index, current in enumerate(times):
        for asset_index, asset in enumerate(assets):
            first = float(asset_index + 1)
            second = float((asset_index + 1) ** 2)
            first_rows.append((current, asset, first))
            second_rows.append((current, asset, second))
            if time_index < 4 and time_index != 1:
                target_rows.append((current, asset, first + second))
                availability_rows.append(
                    (current, asset, float(times[time_index + 1].toordinal()))
                )

    result = (
        OLSPredictionComposer(2)
        .compose(
            _panel(domain, "first", first_rows),
            _panel(domain, "second", second_rows),
            training=PredictionTrainingContext(
                _panel(domain, "targets", target_rows),
                _panel(domain, "availability", availability_rows),
            ),
        )
        .compute(dense_output=False)
        .collect(dense=False)
    )

    assert date(2024, 4, 1) not in result.get_column("time")
    assert result.get_column("time").unique().to_list() == [date(2024, 5, 1)]
