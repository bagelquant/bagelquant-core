import numpy as np
import pandas as pd
import pytest

from bagelquant_core import CategoryPanel, Panel
from bagelquant_core.composer import (
    and_,
    coalesce,
    group_demean,
    greater,
    mask,
    orthogonalize,
    power_df,
    project,
    rolling_corr,
    rolling_ols,
    vol_scale,
)
from bagelquant_core.transformer import (
    anscombe,
    boxcox,
    constant,
    date_age_constraint,
    delta,
    demean,
    denoise,
    fisher,
    inv_log_sqrt_rank,
    kelly_rescaling_weight,
    lag,
    negonly,
    net_scale,
    nonnans,
    normalize,
    notnan,
    nrank,
    posonly,
    rankpct,
    rate_of_change,
    remove_repeated,
    replace_inf,
    rolling_median,
    rolling_percentile,
    rolling_var,
    rolling_zscore,
    sign,
    translate_to_pos,
    trim,
    trim_quantile,
    truncate,
)
from tests.helpers import make_category_panel, make_panel


def panel(values: dict[str, list[float]], *, name: str = "input") -> Panel:
    return make_panel(pd.DataFrame(values), name=name)


def test_general_transformers() -> None:
    source = panel({"a": [1.0, 1.0, np.inf], "b": [np.nan, -1e-14, -3.0]})

    assert nonnans(source).compute().data.iloc[0].tolist() == [1.0, 0.0]
    assert notnan(source).compute().data.iloc[0].tolist() == [1.0, 0.0]
    assert denoise(source).compute().data.iloc[1]["b"] == 0
    assert np.isnan(posonly(source).compute().data.iloc[1]["b"])
    assert np.isnan(negonly(source).compute().data.iloc[0]["a"])
    assert lag(source).compute().data.iloc[0].isna().all()
    assert delta(source).compute().data.iloc[1]["a"] == 0
    assert rate_of_change(panel({"a": [1.0, 2.0, 5.0]}), interval=2).compute().data.iloc[2]["a"] == 2
    assert np.isnan(remove_repeated(source).compute().data.iloc[1]["a"])
    assert constant(source, value=7).compute().data.iloc[0].tolist() == [7, 7]
    assert np.isnan(replace_inf(source).compute().data.iloc[2]["a"])


def test_cross_sectional_transformers() -> None:
    source = panel({"a": [-1.0, 2.0], "b": [1.0, 4.0], "c": [3.0, 6.0]})

    assert demean(source).compute().data.iloc[0].tolist() == [-2.0, 0.0, 2.0]
    assert translate_to_pos(source).compute().data.iloc[0].tolist() == [0.0, 2.0, 4.0]
    assert normalize(source).compute().data.iloc[0].tolist() == [-1.0, 0.0, 1.0]
    assert net_scale(source).compute().data.iloc[0].tolist() == [-1.0, 0.25, 0.75]
    assert rankpct(source).compute().data.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1])
    assert nrank(source).compute().data.iloc[0].tolist() == pytest.approx([-1 / 3, 1 / 3, 1])
    assert np.isnan(net_scale(panel({"a": [np.nan], "b": [1.0]})).compute().data.iloc[0]["a"])


def test_outlier_variance_and_fisher_transformers() -> None:
    source = panel({"a": [-1.0], "b": [0.0], "c": [3.0]})

    assert truncate(source, lower=0, upper=2).compute().data.iloc[0].tolist() == [0, 0, 2]
    assert trim(source, lower=0, upper=2).compute().data.iloc[0][["b"]].tolist() == [0]
    assert trim_quantile(source, lower=0.25, upper=0.75).compute().data.iloc[0]["b"] == 0
    assert anscombe(source).compute().data.iloc[0]["a"] == pytest.approx(2 * np.sqrt(3 / 8))
    assert fisher(panel({"a": [0.5]})).compute().data.iloc[0]["a"] == pytest.approx(np.arctanh(0.5))


def test_boxcox_rank_log_and_kelly_transformers() -> None:
    source = panel({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 8.0]})

    assert boxcox(source).compute().data.iloc[1]["a"] == pytest.approx(np.log(2))
    assert inv_log_sqrt_rank(source).compute().data.iloc[0]["a"] > 0
    weights = kelly_rescaling_weight(source, window=2).compute().data
    assert weights.iloc[1:].ge(0).all().all()
    assert weights.iloc[1:].le(1).all().all()


def test_rolling_catalog() -> None:
    source = panel({"a": [1.0, 2.0, 3.0]})

    assert rolling_median(source, window=2).compute().data["a"].tolist() == pytest.approx(
        [np.nan, 1.5, 2.5], nan_ok=True
    )
    assert rolling_var(source, window=2).compute().data.iloc[1]["a"] == pytest.approx(0.5)
    assert rolling_percentile(source, window=2).compute().data.iloc[2]["a"] == 1
    assert rolling_zscore(source, window=2).compute().data.iloc[2]["a"] == pytest.approx(np.sqrt(0.5))
    assert date_age_constraint(source, window=2).compute().data.iloc[0].isna().all()


def test_general_math_and_scaling_composers() -> None:
    left = panel({"a": [1.0, np.nan], "b": [0.0, 4.0]}, name="left")
    right = panel({"a": [2.0, 3.0], "b": [1.0, 2.0]}, name="right")

    projected = project(left, right).compute().data
    assert np.isnan(projected.iloc[0]["a"])
    assert projected.iloc[0]["b"] == 0
    assert mask(left, right).compute().data.iloc[0].tolist() == [1, 0]
    assert np.isnan(mask(right, left).compute().data.iloc[1]["a"])
    assert coalesce(left, right).compute().data.iloc[1]["a"] == 3
    assert power_df(right, right).compute().data.iloc[0].tolist() == [4, 1]
    assert and_(left, right).compute().data.iloc[0].tolist() == [1, 0]
    assert greater(right, left).compute().data.iloc[0].tolist() == [1, 1]
    assert vol_scale(right, right).compute().data.iloc[0].tolist() == [1, 1]


def test_group_and_rolling_composers() -> None:
    values = panel({"a": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 5.0, 7.0, 9.0]}, name="values")
    groups = make_category_panel(pd.DataFrame({"a": ["x"] * 4, "b": ["x"] * 4}), name="groups")
    factor = panel({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0]}, name="factor")

    assert group_demean(values, groups).compute().data.iloc[0].tolist() == [-1, 1]
    assert rolling_corr(values, factor, window=2).compute().data.iloc[1].tolist() == [1, 1]
    assert rolling_ols(values, factor, window=2).compute().data.iloc[2].tolist() == pytest.approx([3, 7])


def test_orthogonalize_removes_cross_sectional_linear_factor() -> None:
    source = panel({"a": [1.0], "b": [3.0], "c": [5.0]}, name="source")
    factor = panel({"a": [0.0], "b": [1.0], "c": [2.0]}, name="factor")

    assert orthogonalize(source, factor).compute().data.iloc[0].tolist() == pytest.approx([0, 0, 0], abs=1e-12)
