"""Basic element-wise and time-series operators."""

from __future__ import annotations

import polars as pl

from bagelquant_core.frame import ASSET_ID, TIME, VALUE, panel_like, unary
from bagelquant_core.operator._plans import _expression_plan, _ordered_expression_plan
from bagelquant_core.operator._definition import operator


@operator
def identity(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.clone()


@operator(output_type="prediction", input_types=("numeric", "prediction"))
def prediction_signal(frame: pl.DataFrame) -> pl.DataFrame:
    """Declare a numerical signal as a prediction without changing coordinates.

    This explicit typed boundary preserves missing values and availability.
    """
    return frame.clone()


@operator
def negate(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(frame, -pl.col(VALUE))


@operator
def diff(frame: pl.DataFrame, *, periods: int = 1) -> pl.DataFrame:
    _validate_periods(periods, "diff")
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        (pl.col(VALUE) - pl.col(VALUE).shift(periods).over(ASSET_ID)),
    )


@operator
def pct_change(frame: pl.DataFrame, *, periods: int = 1) -> pl.DataFrame:
    return pct_change_frame(frame, periods=periods)


def pct_change_frame(frame: pl.DataFrame, *, periods: int = 1) -> pl.DataFrame:
    """Calculate panel percentage change without constructing a graph."""

    _validate_periods(periods, "pct_change")
    previous = pl.col(VALUE).shift(periods).over(ASSET_ID)
    return panel_like(frame.sort([ASSET_ID, TIME]), pl.col(VALUE) / previous - 1.0)


def _validate_periods(periods: int, operation: str) -> None:
    if not isinstance(periods, int) or isinstance(periods, bool):
        raise TypeError(f"{operation} periods must be an integer")
    if periods <= 0:
        raise ValueError(f"{operation} periods must be positive")


def _plan_diff(
    frame: pl.LazyFrame,
    config: dict[str, object],
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    periods = config.get("periods", 1)
    _validate_periods(periods, "diff")
    return _ordered_expression_plan(
        frame,
        pl.col(VALUE) - pl.col(VALUE).shift(periods).over(ASSET_ID),
        order,
        asset_time_ordered,
    )


def _plan_pct_change(
    frame: pl.LazyFrame,
    config: dict[str, object],
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    periods = config.get("periods", 1)
    _validate_periods(periods, "pct_change")
    previous = pl.col(VALUE).shift(periods).over(ASSET_ID)
    return _ordered_expression_plan(
        frame,
        pl.col(VALUE) / previous - 1.0,
        order,
        asset_time_ordered,
    )


diff._set_plan_operation(  # type: ignore[attr-defined]
    lambda frame, config, order, asset_time_ordered: _plan_diff(
        frame,
        dict(config),
        order,
        asset_time_ordered,
    )
)
pct_change._set_plan_operation(  # type: ignore[attr-defined]
    lambda frame, config, order, asset_time_ordered: _plan_pct_change(
        frame,
        dict(config),
        order,
        asset_time_ordered,
    )
)

identity._set_plan_operation(  # type: ignore[attr-defined]
    lambda frame, config, order, asset_time_ordered: (
        frame.select(TIME, ASSET_ID, VALUE),
        order,
        asset_time_ordered,
    )
)
negate._set_plan_operation(  # type: ignore[attr-defined]
    lambda frame, config, order, asset_time_ordered: _expression_plan(
        frame,
        -pl.col(VALUE),
        order,
        asset_time_ordered,
    )
)
