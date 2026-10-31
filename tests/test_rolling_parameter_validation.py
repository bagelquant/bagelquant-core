"""Lazy rolling execution preserves the eager integer parameter boundary."""
import pytest

from bagelquant_core import operator
from helpers import panel


@pytest.mark.parametrize("name", [
    "rolling_mean", "rolling_std", "rolling_min", "rolling_max", "rolling_sum",
    "rolling_var", "rolling_median", "rolling_skew", "rolling_kurt",
    "rolling_zscore", "rolling_ewm_fw",
])
@pytest.mark.parametrize(("parameter", "invalid"), [
    ("window", 2.9), ("window", True), ("window", "2"),
    ("min_periods", 1.9), ("min_periods", True), ("min_periods", "1"),
])
def test_lazy_rolling_rejects_noninteger_parameters_like_eager(name, parameter, invalid):
    source = panel([
        ("2024-01-02", "A", 1.0),
        ("2024-01-03", "A", 2.0),
        ("2024-01-04", "A", 3.0),
    ])
    operation = getattr(operator, name)
    parameters = {"window": 2, "min_periods": 1, parameter: invalid}
    if name == "rolling_ewm_fw":
        parameters["halflife"] = 1.0
    message = "rolling window" if parameter == "window" else "rolling min_periods"

    with pytest.raises(ValueError, match=message):
        operation.operation(source.collect(dense=True), **parameters)
    with pytest.raises(ValueError, match=message):
        operation(source, **parameters).compute()
