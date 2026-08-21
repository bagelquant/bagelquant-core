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
    ICWeightedDecayPredictionComposer,
    ICWeightedPredictionComposer,
    IdentityPredictionComposer,
    OLSPredictionComposer,
    Panel,
    PredictionPanel,
    PredictionTrainingContext,
    QuantileICWeightedPredictionComposer,
    fama_macbeth_ols_prediction,
    quantile_rank_information_coefficient,
)
from bagelquant_core.composer import add
from bagelquant_core.transformer import group_demean, winsorize, zscore


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


def test_identity_prediction_is_typed_and_round_trips_spec() -> None:
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
    transformed_graph = zscore(winsorize(graph, lower=0.0, upper=1.0))
    transformed = transformed_graph.compute(
        dense_output=False
    )
    assert isinstance(transformed, PredictionPanel)
    assert transformed.collect(dense=False).get_column("value").to_list() == pytest.approx(
        [-2**-0.5, 2**-0.5]
    )
    restored = Graph.compile(transformed_graph.spec().to_dict()).compute(
        {"alpha": alpha}, dense_output=False
    )
    assert isinstance(restored, PredictionPanel)
    rebound = winsorize(result, lower=0.0, upper=1.0).compute(dense_output=False)
    assert isinstance(rebound, PredictionPanel)


def test_prediction_may_only_feed_transformer_semantic_inputs() -> None:
    day = date(2024, 1, 31)
    domain = Domain(calendar=[day], universe=["A", "B"])
    alpha = _panel(domain, "alpha", [(day, "A", 1.0), (day, "B", 3.0)])
    groups = _panel(domain, "groups", [(day, "A", 1.0), (day, "B", 2.0)])
    prediction = IdentityPredictionComposer().compose(alpha)

    with pytest.raises(ValueError, match="semantic input"):
        add(prediction, alpha)
    with pytest.raises(ValueError, match="semantic input"):
        group_demean(groups, group=prediction)


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


def test_ic_weighted_decay_prefers_recent_ic_and_serializes_half_life() -> None:
    times = [date(2024, month, 1) for month in range(1, 4)]
    assets = ["A", "B", "C"]
    domain = Domain(calendar=times, universe=assets)

    def alpha(
        name: str,
        periods: list[list[float | None]],
    ) -> Panel:
        return _panel(
            domain,
            name,
            [
                (current, asset_id, periods[period_index][asset_index])
                for period_index, current in enumerate(times)
                for asset_index, asset_id in enumerate(assets)
            ],
        )

    recent = alpha(
        "recent",
        [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [3.0, None, 3.0]],
    )
    steady = alpha(
        "steady",
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 2.0, None]],
    )
    fading = alpha(
        "fading",
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [100.0, 100.0, 100.0]],
    )
    incomplete = alpha(
        "incomplete",
        [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [100.0, 100.0, 100.0]],
    )
    targets = _panel(
        domain,
        "targets",
        [
            (current, asset_id, float(asset_index + 1))
            for current in times[:2]
            for asset_index, asset_id in enumerate(assets)
        ],
    )
    availability = _panel(
        domain,
        "availability",
        [
            (current, asset_id, float(times[period_index + 1].toordinal()))
            for period_index, current in enumerate(times[:2])
            for asset_id in assets
        ],
    )
    composer = ICWeightedDecayPredictionComposer(window=2, half_life=1)
    graph = composer.compose(
        recent,
        steady,
        fading,
        incomplete,
        training=PredictionTrainingContext(targets, availability),
    )

    result = graph.compute(dense_output=False).collect(dense=False)
    composer_node = next(
        node
        for node in graph.spec().to_dict()["nodes"]
        if node["node_type"] == "prediction_composer"
    )

    # recent has IC [-1, +1], so its half-life-one score is 1/3; steady is 1.
    # fading is clipped at zero and incomplete lacks a finite full window.
    assert result.get_column("time").unique().to_list() == [times[2]]
    assert result.get_column("value").to_list() == pytest.approx([0.75, 2.0, 3.0])
    assert composer_node["config"] == {
        "prediction_composer": "ic_weighted_decay",
        "alpha_count": 4,
        "window": 2,
        "half_life": 1,
    }
    Graph.compile(graph.spec().to_dict())


@pytest.mark.parametrize("half_life", [0, -1, True, 1.5])
def test_ic_weighted_decay_requires_positive_integer_half_life(
    half_life: object,
) -> None:
    with pytest.raises(ValueError, match="half_life"):
        ICWeightedDecayPredictionComposer(window=12, half_life=half_life)  # type: ignore[arg-type]


def test_quantile_rank_ic_uses_complete_high_to_low_group_returns() -> None:
    values = list(range(10, 0, -1))

    assert quantile_rank_information_coefficient(
        values,
        list(range(10, 0, -1)),
        quantiles=10,
    ) == pytest.approx(1.0)
    assert quantile_rank_information_coefficient(
        values,
        list(range(1, 11)),
        quantiles=10,
    ) == pytest.approx(-1.0)
    assert (
        quantile_rank_information_coefficient(
            values,
            [1.0] * 10,
            quantiles=10,
        )
        is None
    )
    assert quantile_rank_information_coefficient(
        [1.0] * 10,
        list(range(10, 0, -1)),
        quantiles=10,
    ) == pytest.approx(1.0)
    assert quantile_rank_information_coefficient(
        [float(20 - index) for index in range(20)],
        [None, 10.0, 9.0, 9.0, 8.0, 8.0, 7.0, 7.0, 6.0, 6.0,
         5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0],
        quantiles=10,
    ) == pytest.approx(1.0)
    assert (
        quantile_rank_information_coefficient(
            values,
            [None, *range(9, 0, -1)],
            quantiles=10,
        )
        is None
    )


def test_quantile_ic_weighted_uses_positive_full_window_and_serializes() -> None:
    times = [
        date(2024 + index // 12, index % 12 + 1, 1) for index in range(13)
    ]
    assets = [f"A{index:02d}" for index in range(10)]
    domain = Domain(calendar=times, universe=assets)
    positive = [
        (current, asset, float(10 - index))
        for current in times
        for index, asset in enumerate(assets)
    ]
    negative = [
        (current, asset, float(index + 1))
        for current in times
        for index, asset in enumerate(assets)
    ]
    targets = [
        (current, asset, float(10 - index))
        for current in times[:12]
        for index, asset in enumerate(assets)
    ]
    availability = [
        (current, asset, float(times[time_index + 1].toordinal()))
        for time_index, current in enumerate(times[:12])
        for asset in assets
    ]
    composer = QuantileICWeightedPredictionComposer(window=12, quantiles=10)
    graph = composer.compose(
        _panel(domain, "positive", positive),
        _panel(domain, "negative", negative),
        training=PredictionTrainingContext(
            _panel(domain, "targets", targets),
            _panel(domain, "availability", availability),
        ),
    )

    result = graph.compute(dense_output=False).collect(dense=False)
    composer_node = next(
        node
        for node in graph.spec().to_dict()["nodes"]
        if node["node_type"] == "prediction_composer"
    )

    assert result.get_column("time").unique().to_list() == [times[12]]
    assert result.get_column("value").to_list() == pytest.approx(
        [float(10 - index) for index in range(10)]
    )
    assert composer_node["config"] == {
        "prediction_composer": "quantile_ic_weighted",
        "alpha_count": 2,
        "window": 12,
        "quantiles": 10,
    }
    Graph.compile(graph.spec().to_dict())


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


def test_fama_macbeth_ols_exposes_same_prediction_and_diagnostics() -> None:
    times = [date(2024, month, 1) for month in range(1, 4)]
    assets = ["A", "B", "C", "D", "E"]
    domain = Domain(calendar=times, universe=assets)
    first_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    second_values = [0.0, 1.0, 4.0, 1.0, 0.0]
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
    target_rows: list[tuple[date, str, float]] = []
    availability_rows: list[tuple[date, str, float]] = []
    coefficients = ((0.1, 0.2, -0.3), (0.3, 0.4, -0.1))
    for period_index, current in enumerate(times[:2]):
        intercept, first_beta, second_beta = coefficients[period_index]
        for asset_index, asset in enumerate(assets):
            target_rows.append(
                (
                    current,
                    asset,
                    intercept
                    + first_beta * first_values[asset_index]
                    + second_beta * second_values[asset_index],
                )
            )
            availability_rows.append(
                (current, asset, float(times[period_index + 1].toordinal()))
            )
    first = _panel(domain, "first", first_rows)
    second = _panel(domain, "second", second_rows)
    targets = _panel(domain, "targets", target_rows)
    availability = _panel(domain, "availability", availability_rows)

    diagnostic = fama_macbeth_ols_prediction(
        {
            "first_factor": first.collect(dense=False),
            "second_factor": second.collect(dense=False),
        },
        targets.collect(dense=False),
        availability.collect(dense=False),
        window=2,
    )
    composer_prediction = (
        OLSPredictionComposer(2)
        .compose(
            first,
            second,
            training=PredictionTrainingContext(targets, availability),
        )
        .compute(dense_output=False)
        .collect(dense=False)
    )

    assert diagnostic.prediction.equals(composer_prediction)
    assert diagnostic.factor_returns.get_column("factor").unique().sort().to_list() == [
        "first_factor",
        "intercept",
        "second_factor",
    ]
    assert diagnostic.rolling_premia.filter(
        pl.col("time") == times[2]
    ).select("factor", "mean_coefficient").sort("factor").to_dicts() == [
        {"factor": "first_factor", "mean_coefficient": pytest.approx(0.3)},
        {"factor": "intercept", "mean_coefficient": pytest.approx(0.2)},
        {"factor": "second_factor", "mean_coefficient": pytest.approx(-0.2)},
    ]
    assert diagnostic.period_diagnostics.get_column("status").to_list() == [
        "complete",
        "complete",
    ]
    assert diagnostic.period_diagnostics.get_column("rank").to_list() == [3, 3]


def test_fama_macbeth_ols_reports_rank_deficiency() -> None:
    times = [date(2024, 1, 1), date(2024, 2, 1)]
    assets = ["A", "B", "C", "D", "E"]
    first = pl.DataFrame(
        {
            "time": [times[0]] * len(assets) + [times[1]] * len(assets),
            "asset_id": assets * 2,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        }
    )
    second = first.with_columns((pl.col("value") * 2.0).alias("value"))
    targets = first.filter(pl.col("time") == times[0])
    availability = targets.with_columns(
        pl.lit(float(times[1].toordinal())).alias("value")
    )

    result = fama_macbeth_ols_prediction(
        {"first": first, "second": second},
        targets,
        availability,
        window=1,
    )

    assert result.prediction.is_empty()
    assert result.factor_returns.is_empty()
    assert result.period_diagnostics.row(0, named=True)["status"] == "rank_deficient"


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
