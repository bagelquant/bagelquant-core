"""Shared one-line documentation for public panel operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


OPERATION_DESCRIPTIONS = {
    "abs": "Return the absolute value of each element.",
    "and_": "Return one where both corresponding elements are truthy and zero elsewhere.",
    "arccos": "Return the inverse cosine of each element, masking values outside `[-1, 1]`.",
    "arcsin": "Return the inverse sine of each element, masking values outside `[-1, 1]`.",
    "arctan": "Return the inverse tangent of each element.",
    "arctanh": "Return the inverse hyperbolic tangent of each element, masking values outside `(-1, 1)`.",
    "coalesce": "Return the first non-missing value from the supplied inputs for each cell.",
    "cos": "Return the cosine of each element.",
    "divide": "Alias for `div`. Divide the first input by the second element-wise.",
    "equal": "Return one where corresponding elements are equal and zero elsewhere.",
    "greater": "Return one where the first input is greater than the second and zero elsewhere.",
    "greater_equal": "Return one where the first input is greater than or equal to the second and zero elsewhere.",
    "group_demean": "Subtract the row-wise group mean from each element.",
    "group_max": "Replace each element with its row-wise group maximum.",
    "group_mean": "Replace each element with its row-wise group mean.",
    "group_median": "Replace each element with its row-wise group median.",
    "group_min": "Replace each element with its row-wise group minimum.",
    "group_percentile": "Return row-wise percentile ranks within each group.",
    "group_rank": "Return row-wise ranks within each group.",
    "group_rankpct": "Return row-wise dense percentile ranks within each group.",
    "group_std": "Replace each element with its row-wise group standard deviation.",
    "group_zscore": "Return row-wise z-scores within each group.",
    "date_age_constraint": "Mask values until a trailing window contains the required number of valid observations.",
    "inv_log_sqrt_rank": "Return `-log(rank) / sqrt(rank)` using cross-sectional percentile ranks.",
    "less": "Return one where the first input is less than the second and zero elsewhere.",
    "less_equal": "Return one where the first input is less than or equal to the second and zero elsewhere.",
    "max": "Alias for `maximum`. Return element-wise maximum values.",
    "min": "Alias for `minimum`. Return element-wise minimum values.",
    "multiply": "Alias for `mul`. Multiply two inputs element-wise.",
    "net_scale": "Scale positive and negative cross-sectional values independently by their absolute-side sums.",
    "nonnans": "Replace null and NaN values with zero.",
    "normalize": "Scale each cross-section linearly to `[-1, 1]`.",
    "notnan": "Return one for finite or infinite present values and zero for null or NaN values.",
    "not_": "Return one where elements are falsy and zero where they are truthy.",
    "or_": "Return one where either corresponding element is truthy and zero elsewhere.",
    "power": "Raise each element of the first input to the corresponding element of the second input.",
    "power_df": "Raise each element of the first input to the corresponding element of the second input.",
    "rate_of_change": "Return the value difference over `interval` rows divided by that interval.",
    "rolling_elastic_net": "Predict the current target from factors fitted on the prior window with elastic-net regularization.",
    "rolling_lasso": "Predict the current target from factors fitted on the prior window with L1 regularization.",
    "rolling_ols": "Predict the current target from one or more factors fitted on the prior window.",
    "rolling_ridge": "Predict the current target from factors fitted on the prior window with L2 regularization.",
    "rolling_ew_std": "Alias for `ewm_std`. Return exponentially weighted standard deviations.",
    "rolling_ewm": "Alias for `ewm_mean`. Return exponentially weighted means.",
    "sin": "Return the sine of each element.",
    "subtract": "Alias for `sub`. Subtract the second input from the first element-wise.",
    "xand": "Return one where corresponding truth values are equivalent and zero elsewhere.",
    "xor": "Return one where exactly one corresponding element is truthy and zero elsewhere.",
}


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
    "operation_description",
]
