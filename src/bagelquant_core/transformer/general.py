"""General-purpose transformers."""

from __future__ import annotations

from numbers import Real

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, panel_like, unary
from .core import _expression_plan, _ordered_expression_plan, transformer


@transformer
def notnan(frame: pl.DataFrame) -> pl.DataFrame:
    present = pl.col(VALUE).is_not_null() & ~pl.col(VALUE).is_nan()
    return unary(frame, present.cast(pl.Float64))


@transformer
def denoise(frame: pl.DataFrame, *, threshold: float = 1e-12) -> pl.DataFrame:
    if not isinstance(threshold, Real) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("denoise threshold must be a non-negative real number")
    return unary(
        frame,
        pl.when(pl.col(VALUE).abs() < threshold).then(0.0).otherwise(pl.col(VALUE)),
    )


@transformer
def posonly(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(
        frame, pl.when(pl.col(VALUE) >= 0).then(pl.col(VALUE)).otherwise(None)
    )


@transformer
def negonly(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(
        frame, pl.when(pl.col(VALUE) <= 0).then(pl.col(VALUE)).otherwise(None)
    )


@transformer
def lag(frame: pl.DataFrame, *, periods: int = 1) -> pl.DataFrame:
    _validate_periods(periods, operation="lag")
    return panel_like(
        frame.sort([ASSET_ID, TIME]), pl.col(VALUE).shift(periods).over(ASSET_ID)
    )


@transformer
def remove_repeated(frame: pl.DataFrame) -> pl.DataFrame:
    previous = pl.col(VALUE).shift(1).over(ASSET_ID)
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.when(pl.col(VALUE) == previous).then(None).otherwise(pl.col(VALUE)),
    )


@transformer
def date_age_constraint(
    frame: pl.DataFrame,
    *,
    window: int,
    min_valid: int | None = None,
) -> pl.DataFrame:
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("date_age_constraint window must be a positive integer")
    required = window if min_valid is None else min_valid
    if (
        not isinstance(required, int)
        or isinstance(required, bool)
        or required <= 0
        or required > window
    ):
        raise ValueError("date_age_constraint min_valid must be in [1, window]")
    valid_count = (
        (pl.col(VALUE).is_not_null() & ~pl.col(VALUE).is_nan())
        .cast(pl.Int64)
        .rolling_sum(window, min_samples=1)
        .over(ASSET_ID)
    )
    return panel_like(
        frame.sort([ASSET_ID, TIME]),
        pl.when(valid_count >= required).then(pl.col(VALUE)).otherwise(None),
    )


@transformer
def constant(frame: pl.DataFrame, *, value: float = 1) -> pl.DataFrame:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("constant value must be a real number")
    return unary(frame, pl.lit(float(value)))


@transformer
def replace_inf(frame: pl.DataFrame) -> pl.DataFrame:
    return unary(
        frame, pl.when(pl.col(VALUE).is_infinite()).then(None).otherwise(pl.col(VALUE))
    )


def _validate_periods(periods: int, *, operation: str) -> None:
    if not isinstance(periods, int) or isinstance(periods, bool):
        raise TypeError(f"{operation} periods must be an integer")
    if periods <= 0:
        raise ValueError(f"{operation} periods must be positive")


def _plan_general_time_series(
    name: str,
    frame: pl.LazyFrame,
    config: dict[str, object],
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    if name == "remove_repeated":
        previous = pl.col(VALUE).shift(1).over(ASSET_ID)
        expression = (
            pl.when(pl.col(VALUE) == previous)
            .then(None)
            .otherwise(pl.col(VALUE))
        )
    elif name == "date_age_constraint":
        window = config.get("window")
        if (
            not isinstance(window, int)
            or isinstance(window, bool)
            or window <= 0
        ):
            raise ValueError(
                "date_age_constraint window must be a positive integer"
            )
        raw_required = config.get("min_valid")
        required = window if raw_required is None else raw_required
        if (
            not isinstance(required, int)
            or isinstance(required, bool)
            or required <= 0
            or required > window
        ):
            raise ValueError(
                "date_age_constraint min_valid must be in [1, window]"
            )
        valid_count = (
            (pl.col(VALUE).is_not_null() & ~pl.col(VALUE).is_nan())
            .cast(pl.Int64)
            .rolling_sum(window, min_samples=1)
            .over(ASSET_ID)
        )
        expression = (
            pl.when(valid_count >= required)
            .then(pl.col(VALUE))
            .otherwise(None)
        )
    else:
        raise ValueError(f"unsupported time-series plan operation: {name}")
    return _ordered_expression_plan(
        frame,
        expression,
        order,
        asset_time_ordered,
    )


for _plan_name, _plan_transformer in {
    "remove_repeated": remove_repeated,
    "date_age_constraint": date_age_constraint,
}.items():
    _plan_transformer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frame, config, order, asset_time_ordered, name=_plan_name: (
            _plan_general_time_series(
                name,
                frame,
                dict(config),
                order,
                asset_time_ordered,
            )
        )
    )


def _plan_general_pointwise(
    name: str,
    frame: pl.LazyFrame,
    config: dict[str, object],
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    value = pl.col(VALUE)
    if name == "notnan":
        expression = (value.is_not_null() & ~value.is_nan()).cast(pl.Float64)
    elif name == "denoise":
        threshold = config.get("threshold", 1e-12)
        if (
            not isinstance(threshold, Real)
            or isinstance(threshold, bool)
            or threshold < 0
        ):
            raise ValueError("denoise threshold must be a non-negative real number")
        expression = (
            pl.when(value.abs() < float(threshold)).then(0.0).otherwise(value)
        )
    elif name == "posonly":
        expression = pl.when(value >= 0).then(value).otherwise(None)
    elif name == "negonly":
        expression = pl.when(value <= 0).then(value).otherwise(None)
    elif name == "constant":
        scalar = config.get("value", 1)
        if not isinstance(scalar, Real) or isinstance(scalar, bool):
            raise TypeError("constant value must be a real number")
        expression = pl.lit(float(scalar))
    else:
        expression = pl.when(value.is_infinite()).then(None).otherwise(value)
    return _expression_plan(
        frame,
        expression,
        order,
        asset_time_ordered,
    )


for _plan_name, _plan_transformer in {
    "notnan": notnan,
    "denoise": denoise,
    "posonly": posonly,
    "negonly": negonly,
    "constant": constant,
    "replace_inf": replace_inf,
}.items():
    _plan_transformer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frame, config, order, asset_time_ordered, name=_plan_name: (
            _plan_general_pointwise(
                name,
                frame,
                dict(config),
                order,
                asset_time_ordered,
            )
        )
    )
