from datetime import date

import numpy as np
import polars as pl
import pytest

from bagelquant_core import Domain, Node
from bagelquant_core.operator import cross_sectional_residual, cross_sectional_weighted_mean, mad_zscore
from bagelquant_core.operator import log1p
from bagelquant_core.operator.robust import mad_standardize_values, ols_residual_values


@pytest.mark.parametrize("seed", range(5))
def test_population_mad_and_ols_match_independent_legacy_formulas(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=71)
    values[:4] = [np.nan, np.inf, -np.inf, 1e9]
    valid = np.isfinite(values)
    expected = np.full(values.shape, np.nan)
    sample = values[valid].copy()
    median = np.median(sample)
    mad = np.median(np.abs(sample-median))
    sample = np.clip(sample, median-5*1.4826*mad, median+5*1.4826*mad)
    expected[valid] = (sample-sample.mean())/sample.std(ddof=0)
    np.testing.assert_array_equal(mad_standardize_values(values), expected)
    controls = rng.normal(size=(71, 2))
    controls[5, 1] = np.nan
    finite = np.isfinite(values) & np.isfinite(controls).all(axis=1)
    design = np.column_stack([np.ones(finite.sum()), controls[finite]])
    residual = np.full(values.shape, np.nan)
    residual[finite] = values[finite] - design@np.linalg.lstsq(design, values[finite], rcond=None)[0]
    np.testing.assert_array_equal(ols_residual_values(values, controls), residual)


def test_degenerate_crosssections_are_missing_and_typed_nodes_propagate_all_asset_traces():
    assert np.isnan(mad_standardize_values([1., 1., np.nan])).all()
    assert np.isnan(ols_residual_values([1., 2., 3.], np.ones((3, 1)))).all()
    day = date(2024, 1, 4)
    domain = Domain(calendar=[day], universe=["a", "b", "c", "d"])
    def value(values):
        return Node.from_domain(pl.DataFrame({"time": [day]*4, "asset_id": ["a", "b", "c", "d"],
            "value": values, "available_date": [date(2024, 1, 1), date(2024, 1, 2), day, day]}),
            domain, trace_columns=("available_date",))
    target, controls = value([1., 3., 4., 8.]), value([0., 1., 3., 4.])
    normalized = mad_zscore(target).compute().collect(include_traces=True)
    assert normalized["available_date"].to_list() == [day]*4
    residual = cross_sectional_residual(target, controls=(controls,)).compute().collect()["value"].to_numpy()
    np.testing.assert_array_equal(residual, ols_residual_values([1., 3., 4., 8.], np.array([[0.], [1.], [3.], [4.]])))
    weights = value([1., -1., 4., np.inf])
    mean = cross_sectional_weighted_mean(target, weights=weights).compute().collect()
    assert mean["value"].to_list() == [3.4]*4


def test_log1p_tiny_values_preserve_stable_eager_and_lazy_formula():
    values = [1e-18, -1e-18, 1e-13, -1e-13, 0., -1., -2., None]
    day = date(2024, 1, 4)
    assets = [str(index) for index in range(len(values))]
    frame = pl.DataFrame({"time": [day]*len(values), "asset_id": assets, "value": values})
    expected = frame.with_columns(pl.when(pl.col("value") > -1).then(pl.col("value").log1p()).otherwise(None))
    from polars.testing import assert_frame_equal
    assert_frame_equal(log1p.operation(frame), expected, check_exact=True)
    value = Node.from_domain(frame, Domain(calendar=[day], universe=assets))
    assert_frame_equal(log1p(value).compute().collect(), expected, check_exact=True)
    assert log1p.version == "2"
    assert expected["value"][0] == 1e-18
