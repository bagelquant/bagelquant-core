from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import (
    Domain,
    Panel,
    PredictionPanel,
    PredictionSmoothingConfig,
    PredictionSmoothingState,
    smooth_prediction,
)


def _panel(values, days=None, inactive=()):
    days = days or [date(2024, 1, day + 1) for day in range(len(values))]
    membership = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["a"] * len(days),
            "active": [day not in inactive for day in days],
        }
    )
    return PredictionPanel.from_domain(
        pl.DataFrame({"time": days, "asset_id": ["a"] * len(days), "value": values}),
        Domain(calendar=days, universe=membership),
    )


@pytest.mark.parametrize("method", ["none", "sma", "ewma"])
def test_resume_exactly_matches_full_history_and_serializes(method):
    config = PredictionSmoothingConfig(
        method,
        window=3 if method == "sma" else None,
        half_life=2 if method == "ewma" else None,
    )
    panel = _panel([1.0, 2.0, 4.0, 8.0, None, 16.0, 32.0, 64.0])
    days = panel.domain.times.to_list()
    full = smooth_prediction(panel, config, evaluation_calendar=days)
    for split in range(1, len(days)):
        source = panel.collect(dense=True)
        left = _panel(source["value"].to_list()[:split], days[:split])
        right = _panel(source["value"].to_list()[split:], days[split:])
        first = smooth_prediction(left, config, evaluation_calendar=days)
        restored = PredictionSmoothingState.from_dict(first.state.to_dict())
        second = smooth_prediction(
            right, config, evaluation_calendar=days, state=restored
        )
        assert_frame_equal(
            full.prediction.collect(),
            pl.concat([first.prediction.collect(), second.prediction.collect()]),
        )
        assert second.state == full.state


def test_sma_requires_contiguous_finite_active_observations():
    panel = _panel(
        [1.0, 3.0, None, 7.0, 9.0, float("nan"), 11.0, 13.0, float("inf"), 17.0, 19.0]
    )
    result = smooth_prediction(
        panel,
        PredictionSmoothingConfig("sma", window=2),
        evaluation_calendar=panel.domain.times,
    )
    assert result.prediction.collect()["value"].to_list() == [
        None,
        2.0,
        None,
        None,
        8.0,
        None,
        None,
        12.0,
        None,
        None,
        18.0,
    ]


def test_ewma_initialization_gap_reset_and_universe_exit():
    days = [date(2024, 1, i) for i in range(1, 7)]
    panel = _panel([2.0, 4.0, 100.0, 8.0, None, 16.0], days, inactive=[days[2]])
    result = smooth_prediction(
        panel, PredictionSmoothingConfig("ewma", half_life=1), evaluation_calendar=days
    )
    assert result.prediction.collect()["value"].to_list() == [2.0, 3.0, 8.0, None, 16.0]


def test_monthly_windows_are_observation_periods_and_prefix_causal():
    days = [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 29)]
    config = PredictionSmoothingConfig("sma", window=2)
    full = smooth_prediction(
        _panel([2.0, 4.0, 100.0], days), config, evaluation_calendar=days
    )
    assert full.prediction.collect()["value"].to_list() == [None, 3.0, 52.0]
    short = smooth_prediction(
        _panel([2.0, 4.0], days[:2]), config, evaluation_calendar=days[:2]
    )
    assert_frame_equal(full.prediction.collect().head(2), short.prediction.collect())


def test_calendar_gaps_and_ui_window_truncation_are_rejected():
    days = [date(2024, 1, i) for i in range(1, 5)]
    config = PredictionSmoothingConfig()
    with pytest.raises(ValueError, match="warm-up"):
        smooth_prediction(
            _panel([1.0, 2.0], days[2:]), config, evaluation_calendar=days
        )
    with pytest.raises(ValueError, match="contiguous"):
        smooth_prediction(
            _panel([1.0, 2.0], [days[0], days[2]]), config, evaluation_calendar=days
        )
    with pytest.raises(ValueError, match="ordered unique"):
        smooth_prediction(
            _panel([1.0, 2.0], days[:2]),
            config,
            evaluation_calendar=list(reversed(days)),
        )


def test_checkpoint_rejects_config_or_historical_calendar_changes():
    days = [date(2024, 1, i) for i in range(1, 5)]
    config = PredictionSmoothingConfig()
    first = smooth_prediction(
        _panel([1.0, 2.0], days[:2]), config, evaluation_calendar=days
    )
    with pytest.raises(ValueError, match="configuration"):
        smooth_prediction(
            _panel([3.0, 4.0], days[2:]),
            PredictionSmoothingConfig("sma", window=2),
            evaluation_calendar=days,
            state=first.state,
        )
    with pytest.raises(ValueError, match="history"):
        smooth_prediction(
            _panel([3.0, 4.0], days[2:]),
            config,
            evaluation_calendar=days[1:],
            state=first.state,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "bad"},
        {"method": "sma", "window": True},
        {"method": "sma", "window": 0},
        {"method": "ewma", "half_life": float("inf")},
        {"method": "none", "window": 2},
        {"method": "sma", "window": 2, "half_life": 1},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PredictionSmoothingConfig(**kwargs)


def test_typed_boundary_and_no_input_mutation():
    panel = _panel([1.0, 2.0])
    before = panel.collect()
    result = smooth_prediction(
        panel, PredictionSmoothingConfig(), evaluation_calendar=panel.domain.times
    )
    assert isinstance(result.prediction, PredictionPanel)
    assert_frame_equal(before, result.prediction.collect())
    assert_frame_equal(before, panel.collect())
    with pytest.raises(TypeError, match="PredictionPanel"):
        smooth_prediction(
            Panel.from_domain(before, panel.domain),
            PredictionSmoothingConfig(),
            evaluation_calendar=panel.domain.times,
        )
