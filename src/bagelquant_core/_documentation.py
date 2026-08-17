"""Shared one-line documentation for public panel operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


OPERATION_DESCRIPTIONS = {
    "abs": "Return the absolute value of each element.",
    "abs_value": "Return the absolute value of each element while preserving panel keys.",
    "add": "Add two key-aligned panel values element-wise.",
    "and_": "Return one where both corresponding elements are truthy and zero elsewhere.",
    "anscombe": "Apply the Anscombe square-root transform to stabilize count-data variance.",
    "arccos": "Return the inverse cosine of each element, masking values outside `[-1, 1]`.",
    "arcsin": "Return the inverse sine of each element, masking values outside `[-1, 1]`.",
    "arctan": "Return the inverse tangent of each element.",
    "arctanh": "Return the inverse hyperbolic tangent of each element, masking values outside `(-1, 1)`.",
    "bfill": "Fill each asset's missing rows from later observations, ordered by time and bounded by the finite limit.",
    "boxcox": "Apply the Box-Cox power transform element-wise with the supplied lambda.",
    "coalesce": "Return the first non-missing value from the supplied inputs for each cell.",
    "constant": "Replace every present keyed value with the configured constant while preserving missing membership.",
    "cos": "Return the cosine of each element.",
    "date_age_constraint": "Keep a value only when its per-asset trailing window contains the required number of valid observations.",
    "delta": "Subtract the value `interval` observations earlier within each asset ordered by time.",
    "demean": "Subtract the same-date cross-sectional mean from every present value.",
    "denoise": "Set same-date cross-sectional deviations below the configured magnitude threshold to zero.",
    "diff": "Return the per-asset difference from the observation `periods` rows earlier.",
    "div": "Divide the first key-aligned panel by the second element-wise.",
    "equal": "Return one where corresponding elements are equal and zero elsewhere.",
    "ewm_mean": "Return the per-asset exponentially weighted moving mean in time order.",
    "ewm_std": "Return the per-asset exponentially weighted moving standard deviation in time order.",
    "ewm_var": "Return the per-asset exponentially weighted moving variance in time order.",
    "ffill": "Fill each asset's missing rows from earlier observations, ordered by time and bounded by the finite limit.",
    "fillna": "Replace missing panel values with the configured scalar.",
    "fillna_zero": "Replace null and NaN panel values with zero.",
    "fisher": "Apply the Fisher transform inside `(-1, 1)` and mask invalid inputs.",
    "freeman": "Apply the Freeman-Tukey square-root transform to stabilize count-data variance.",
    "greater": "Return one where the first input is greater than the second and zero elsewhere.",
    "greater_equal": "Return one where the first input is greater than or equal to the second and zero elsewhere.",
    "group_demean": "Subtract the same-date mean of each element's declared group.",
    "group_max": "Replace each value with its declared group's same-date maximum.",
    "group_mean": "Replace each value with its declared group's same-date arithmetic mean.",
    "group_median": "Replace each value with its declared group's same-date median.",
    "group_min": "Replace each value with its declared group's same-date minimum.",
    "group_percentile": "Return average-tie percentile ranks within each declared group and date.",
    "group_rank": "Return average-tie ordinal ranks within each declared group and date.",
    "group_rankpct": "Return dense percentile ranks within each declared group and date.",
    "group_std": "Replace each value with its declared group's same-date standard deviation.",
    "group_zscore": "Standardize values within each declared group and date.",
    "identity": "Return an unchanged copy of the keyed panel values.",
    "inv_log_sqrt_rank": "Return `-log(rank) / sqrt(rank)` using cross-sectional percentile ranks.",
    "kelly": "Estimate a rolling Kelly-style signal from each asset's trailing mean and variance.",
    "kelly_nonan_standardize": "Standardize valid Kelly-style values cross-sectionally without filling missing rows.",
    "kelly_rank_boxcox": "Rank Kelly-style values cross-sectionally and apply a Box-Cox transform.",
    "kelly_rescaling_weight": "Convert Kelly-style values into cross-sectional rescaling weights.",
    "lag": "Return the value `periods` earlier within each asset ordered by time.",
    "less": "Return one where the first input is less than the second and zero elsewhere.",
    "less_equal": "Return one where the first input is less than or equal to the second and zero elsewhere.",
    "log": "Return the natural logarithm of positive values and mask invalid inputs.",
    "log1p": "Return `log(1 + value)` for values greater than `-1` and mask invalid inputs.",
    "log_rank": "Apply a negative logarithm to same-date cross-sectional percentile ranks.",
    "logrank": "Return a logarithmically transformed same-date cross-sectional rank.",
    "mask": "Keep source values where the key-aligned mask is truthy and use the configured replacement elsewhere.",
    "maximum": "Return the element-wise maximum across the aligned input panels.",
    "mean": "Return the element-wise arithmetic mean across aligned input panels.",
    "min_max_scale": "Scale each same-date cross-section linearly into the configured range.",
    "minimum": "Return the element-wise minimum across the aligned input panels.",
    "mul": "Multiply two key-aligned panel values element-wise.",
    "negate": "Negate every present panel value element-wise.",
    "negonly": "Keep negative values and suppress non-negative values.",
    "net_scale": "Scale positive and negative cross-sectional values independently by their absolute-side sums.",
    "non_nan_to_one": "Replace every non-missing value with one while retaining missing rows.",
    "non_nan_to_zero": "Replace every non-missing value with zero while retaining missing rows.",
    "nonnans": "Replace null and NaN values with zero.",
    "normalize": "Scale each cross-section linearly to `[-1, 1]`.",
    "not_": "Return one where elements are falsy and zero where they are truthy.",
    "notnan": "Return one for present values and zero for null or NaN values.",
    "nrank": "Return a normalized same-date cross-sectional rank centered around zero.",
    "or_": "Return one where either corresponding element is truthy and zero elsewhere.",
    "orthogonalize": "Return same-date cross-sectional residuals after regressing the source on the named factor Panels.",
    "pct_change": "Return each asset's percentage change from the observation `periods` rows earlier.",
    "posonly": "Keep positive values and suppress non-positive values.",
    "power": "Raise each present source value to the configured scalar exponent.",
    "power_df": "Raise each element of the first input to the corresponding element of the second input.",
    "product": "Return the element-wise product across all aligned input panels.",
    "project": "Keep source values only where the key-aligned binary Panel equals one.",
    "rank": "Return average-tie ranks within each date cross-section while preserving missing membership.",
    "rankpct": "Return dense percentile ranks within each date cross-section.",
    "rate_of_change": "Return the value difference over `interval` observations divided by that interval.",
    "remove_repeated": "Mask per-asset values that repeat the immediately preceding observation.",
    "replace_inf": "Replace positive and negative infinity with missing values.",
    "replace_non_nan": "Replace every non-missing value with the configured scalar.",
    "rolling_corr": "Return the trailing per-asset correlation between two aligned panels.",
    "rolling_cov": "Return the trailing per-asset covariance between two aligned panels.",
    "rolling_elastic_net": "Predict the current target from factors fitted only on prior per-asset rows with elastic-net regularization.",
    "rolling_ewm_fw": "Return a finite-window exponentially weighted per-asset statistic in time order.",
    "rolling_kurt": "Return trailing per-asset kurtosis over the configured observation window.",
    "rolling_lasso": "Predict the current target from factors fitted only on prior per-asset rows with L1 regularization.",
    "rolling_max": "Return the trailing per-asset maximum over the configured observation window.",
    "rolling_mean": "Return the trailing per-asset arithmetic mean over the configured observation window.",
    "rolling_median": "Return the trailing per-asset median over the configured observation window.",
    "rolling_min": "Return the trailing per-asset minimum over the configured observation window.",
    "rolling_ols": "Predict the current target from factors fitted only on prior per-asset rows by ordinary least squares.",
    "rolling_percentile": "Return the current value's percentile within its per-asset trailing window.",
    "rolling_rank": "Return the current value's rank within its per-asset trailing window.",
    "rolling_ridge": "Predict the current target from factors fitted only on prior per-asset rows with L2 regularization.",
    "rolling_skew": "Return trailing per-asset skewness over the configured observation window.",
    "rolling_std": "Return the trailing per-asset standard deviation over the configured observation window.",
    "rolling_sum": "Return the trailing per-asset sum over the configured observation window.",
    "rolling_var": "Return the trailing per-asset variance over the configured observation window.",
    "rolling_zscore": "Standardize each value by its per-asset trailing mean and standard deviation.",
    "sign": "Return `-1`, `0`, or `1` according to each present value's sign.",
    "sin": "Return the sine of each element.",
    "signed_log1p": "Return `sign(value) * log(1 + abs(value))` element-wise.",
    "signed_power": "Raise absolute values to a scalar power and restore their original signs.",
    "sqrt": "Return the square root of non-negative values and mask invalid inputs.",
    "sub": "Subtract the second key-aligned panel from the first element-wise.",
    "sum_frames": "Return the element-wise sum across all aligned input panels.",
    "translate_to_pos": "Shift each same-date cross-section so its minimum present value is zero.",
    "trig": "Apply the configured trigonometric function element-wise.",
    "trim": "Mask values outside the configured absolute bounds.",
    "trim_quantile": "Mask values outside the configured same-date cross-sectional quantiles.",
    "truncate": "Clip values to the configured lower and upper bounds.",
    "vol_scale": "Divide source values by the key-aligned volatility Panel without estimating volatility implicitly.",
    "weighted_mean": "Return the element-wise weighted mean of aligned value and weight panels.",
    "weighted_sum": "Return the element-wise weighted sum of aligned value and weight panels.",
    "winsorize": "Clip values to the configured same-date cross-sectional quantiles.",
    "xand": "Return one where corresponding truth values are equivalent and zero elsewhere.",
    "xor": "Return one where exactly one corresponding element is truthy and zero elsewhere.",
    "zscore": "Subtract the mean and divide by the standard deviation within each date cross-section.",
}


def operation_category(name: str, *, kind: str) -> str:
    """Return the stable functional category shared by generated catalogs."""

    if name.startswith("group_") or name == "orthogonalize":
        return "Group & neutralization"
    if name in {
        "rolling_ols",
        "rolling_ridge",
        "rolling_lasso",
        "rolling_elastic_net",
    }:
        return "Rolling regression"
    if name.startswith(("rolling_", "ewm_")) or name in {
        "lag",
        "diff",
        "delta",
        "pct_change",
        "rate_of_change",
        "remove_repeated",
    }:
        return "Rolling statistics"
    if name in {"mask", "project", "vol_scale"}:
        return "Masking & scaling"
    if name in {
        "fillna",
        "fillna_zero",
        "ffill",
        "bfill",
        "coalesce",
        "nonnans",
        "replace_inf",
    }:
        return "Missing data"
    if name in {
        "and_",
        "or_",
        "not_",
        "xand",
        "xor",
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
    }:
        return "Logical & comparison"
    if kind == "composer" and name in {"add", "sub", "mul", "div", "power"}:
        return "Arithmetic"
    if kind == "composer":
        return "Aggregation"
    if name in {
        "demean",
        "rank",
        "rankpct",
        "zscore",
        "winsorize",
        "normalize",
        "min_max_scale",
        "net_scale",
    }:
        return "Cross-sectional"
    return "Element-wise"


def operation_description(name: str) -> str:
    """Return the curated or deterministic fallback summary for an operation."""

    return OPERATION_DESCRIPTIONS.get(
        name,
        f"Apply `{name}` to long-form panel inputs.",
    )


def ensure_operation_docstring(operation: Callable[..., Any]) -> None:
    """Populate an undocumented operation from the shared reference summary."""

    if not operation.__doc__:
        operation.__doc__ = operation_description(operation.__name__)


__all__ = [
    "OPERATION_DESCRIPTIONS",
    "ensure_operation_docstring",
    "operation_category",
    "operation_description",
]
