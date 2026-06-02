import numpy as np
import pandas as pd
import pytest

from bagelquant_core import CategoryPanel, Panel
from bagelquant_core.transformer import (
    abs_value,
    bfill,
    category_demean,
    category_mean,
    category_rank,
    category_zscore,
    diff,
    ewm_mean,
    ewm_std,
    ewm_var,
    ffill,
    fillna,
    fillna_zero,
    identity,
    log,
    log1p,
    min_max_scale,
    negate,
    non_nan_to_one,
    non_nan_to_zero,
    pct_change,
    power,
    replace_non_nan,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_sum,
    signed_log1p,
    signed_power,
    sqrt,
    winsorize,
)
from tests.helpers import make_category_panel, make_panel


def panel(values: dict[str, list[float]]) -> Panel:
    return make_panel(pd.DataFrame(values), name="input")


def test_basic_transformers() -> None:
    source = panel({"a": [-1.0, 3.0], "b": [2.0, -4.0]})

    assert identity(source).compute().data.equals(source.data)
    assert abs_value(source).compute().data.values.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert negate(source).compute().data.values.tolist() == [
        [1.0, -2.0],
        [-3.0, 4.0],
    ]


def test_diff_runs_over_time_and_stores_periods_in_graph_spec() -> None:
    source = panel({"a": [1.0, 4.0, 10.0]})

    changed = diff(source, periods=2)

    assert changed.compute().data["a"].iloc[:2].isna().all()
    assert changed.compute().data.iloc[2]["a"] == 9.0
    assert changed.spec().nodes[-1].config["periods"] == 2


def test_pct_change_calculates_returns_from_values() -> None:
    source = panel({"a": [100.0, 110.0, 99.0]})

    returns = pct_change(source).compute().data["a"]

    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1:].tolist() == pytest.approx([0.1, -0.1])


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (rolling_mean, [np.nan, 1.5, 2.5]),
        (rolling_sum, [np.nan, 3.0, 5.0]),
        (rolling_min, [np.nan, 1.0, 2.0]),
        (rolling_max, [np.nan, 2.0, 3.0]),
        (rolling_std, [np.nan, np.sqrt(0.5), np.sqrt(0.5)]),
    ],
)
def test_rolling_transformers_run_over_time(operation, expected) -> None:
    source = panel({"a": [1.0, 2.0, 3.0]})

    result = operation(source, window=2).compute().data["a"]

    assert result.tolist() == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_rolling_transformers_reject_invalid_windows(window) -> None:
    source = panel({"a": [1.0, 2.0]})

    with pytest.raises(ValueError, match="positive integer"):
        rolling_mean(source, window=window).compute()


def test_power_transformers() -> None:
    source = panel({"a": [-4.0, 9.0]})

    squared = power(source, exponent=2).compute().data["a"].tolist()
    signed_root = signed_power(source, exponent=0.5).compute().data["a"].tolist()
    root = sqrt(source).compute().data["a"].tolist()

    assert squared == [16.0, 81.0]
    assert signed_root == [-2.0, 3.0]
    assert np.isnan(root[0])
    assert root[1] == 3.0


def test_logarithmic_transformers_define_invalid_value_behavior() -> None:
    source = panel({"a": [-2.0, -1.0, 0.0, 1.0]})

    natural = log(source).compute().data["a"].tolist()
    shifted = log1p(source).compute().data["a"].tolist()
    signed = signed_log1p(source).compute().data["a"].tolist()

    assert all(np.isnan(value) for value in natural[:3])
    assert natural[3] == 0.0
    assert all(np.isnan(value) for value in shifted[:2])
    assert shifted[2:] == [0.0, pytest.approx(np.log(2.0))]
    assert signed == pytest.approx(
        [-np.log(3.0), -np.log(2.0), 0.0, np.log(2.0)]
    )


def test_cross_sectional_normalizers() -> None:
    source = panel({"a": [1.0, 2.0], "b": [2.0, 2.0], "c": [3.0, 2.0]})

    scaled = min_max_scale(source).compute().data
    clipped = winsorize(source, lower=0.25, upper=0.75).compute().data

    assert scaled.iloc[0].tolist() == [0.0, 0.5, 1.0]
    assert scaled.iloc[1].isna().all()
    assert clipped.iloc[0].tolist() == [1.5, 2.0, 2.5]


def test_winsorize_rejects_invalid_quantile_range() -> None:
    source = panel({"a": [1.0], "b": [2.0]})

    with pytest.raises(ValueError, match="0 <= lower <= upper <= 1"):
        winsorize(source, lower=0.9, upper=0.1).compute()


def test_missing_value_transformers() -> None:
    source = panel({"a": [1.0, np.nan, np.nan], "b": [np.nan, 2.0, np.nan]})

    assert fillna(source, value=-1).compute().data.values.tolist() == [
        [1.0, -1.0],
        [-1.0, 2.0],
        [-1.0, -1.0],
    ]
    assert fillna_zero(source).compute().data.values.tolist() == [
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ]
    assert ffill(source).compute().data.iloc[2].tolist() == [1.0, 2.0]
    assert bfill(source).compute().data.iloc[0].tolist() == [1.0, 2.0]


def test_replacement_transformers_preserve_missing_values() -> None:
    source = panel({"a": [1.0, np.nan], "b": [-2.0, 3.0]})

    replaced = replace_non_nan(source, value=7).compute().data
    ones = non_nan_to_one(source).compute().data
    zeros = non_nan_to_zero(source).compute().data

    assert replaced.iloc[0].tolist() == [7.0, 7.0]
    assert np.isnan(replaced.iloc[1]["a"])
    assert ones.iloc[0].tolist() == [1.0, 1.0]
    assert zeros.iloc[0].tolist() == [0.0, 0.0]


def test_ewm_transformers_match_pandas_methods() -> None:
    source = panel({"a": [1.0, 2.0, 3.0, 4.0]})
    frame = source.data

    averaged = ewm_mean(source, span=3).compute().data
    deviation = ewm_std(source, halflife=2).compute().data
    variance = ewm_var(source, alpha=0.5).compute().data

    assert averaged.equals(frame.ewm(span=3).mean())
    assert deviation.equals(frame.ewm(halflife=2).std())
    assert variance.equals(frame.ewm(alpha=0.5).var())


def test_ewm_requires_exactly_one_decay_argument() -> None:
    source = panel({"a": [1.0, 2.0]})

    with pytest.raises(ValueError, match="exactly one"):
        ewm_mean(source).compute()

    with pytest.raises(ValueError, match="exactly one"):
        ewm_mean(source, span=3, alpha=0.5).compute()


def test_category_panel_supports_string_labels() -> None:
    categories = make_category_panel(
        pd.DataFrame({"a": ["tech"], "b": ["finance"]}),
        name="industry",
    )

    assert categories.data.iloc[0].tolist() == ["tech", "finance"]


def test_category_transformers_operate_within_each_row_group() -> None:
    source = panel(
        {
            "a": [1.0, 8.0],
            "b": [3.0, 4.0],
            "c": [10.0, 2.0],
            "d": [14.0, 6.0],
        }
    )
    categories = make_category_panel(
        pd.DataFrame(
            {
                "a": ["tech", "tech"],
                "b": ["tech", "finance"],
                "c": ["finance", "finance"],
                "d": ["finance", "tech"],
            }
        ),
        name="industry",
    )

    demeaned = category_demean(source, categories).compute().data
    grouped_mean = category_mean(source, categories).compute().data
    ranked = category_rank(source, categories).compute().data
    scored = category_zscore(source, categories).compute().data

    assert demeaned.iloc[0].tolist() == [-1.0, 1.0, -2.0, 2.0]
    assert grouped_mean.iloc[0].tolist() == [2.0, 2.0, 12.0, 12.0]
    assert ranked.iloc[0].tolist() == [0.5, 1.0, 0.5, 1.0]
    assert scored.iloc[0].tolist() == pytest.approx(
        [-np.sqrt(0.5), np.sqrt(0.5), -np.sqrt(0.5), np.sqrt(0.5)]
    )
