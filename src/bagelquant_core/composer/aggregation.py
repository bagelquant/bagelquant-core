"""Aggregation composers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import polars as pl

from ..frame import VALUE, nary, panel_like
from .core import _horizontal_value_plan, composer


@composer
def sum_frames(*frames: pl.DataFrame) -> pl.DataFrame:
    return nary(frames, lambda values: sum(values[1:], values[0]))


@composer
def mean(*frames: pl.DataFrame) -> pl.DataFrame:
    return nary(frames, lambda values: sum(values[1:], values[0]) / len(values))


@composer
def product(*frames: pl.DataFrame) -> pl.DataFrame:
    return nary(
        frames,
        lambda values: pl.fold(
            acc=pl.lit(1.0), function=lambda acc, x: acc * x, exprs=values
        ),
    )


@composer
def minimum(*frames: pl.DataFrame) -> pl.DataFrame:
    return nary(frames, lambda values: pl.min_horizontal(values))


@composer
def maximum(*frames: pl.DataFrame) -> pl.DataFrame:
    return nary(frames, lambda values: pl.max_horizontal(values))


@composer
def weighted_sum(*frames: pl.DataFrame, weights: Sequence[float]) -> pl.DataFrame:
    if not frames:
        raise ValueError("weighted_sum requires at least one frame")
    if len(frames) != len(weights):
        raise ValueError("weights length must match frame count")
    if any(not isinstance(weight, Real) or isinstance(weight, bool) for weight in weights):
        raise TypeError("weights must be real numbers")
    return nary(
        frames,
        lambda values: sum(
            (
                value * float(weight)
                for value, weight in zip(values, weights, strict=True)
            ),
            pl.lit(0.0),
        ),
    )


@composer
def weighted_mean(*frames: pl.DataFrame, weights: Sequence[float]) -> pl.DataFrame:
    total = float(sum(weights))
    if total == 0:
        raise ValueError("weights must not sum to zero")
    return panel_like(
        weighted_sum.operation(*frames, weights=weights), pl.col(VALUE) / total
    )


def _plan_aggregation(
    frames: tuple[pl.LazyFrame, ...],
    config: Mapping[str, object],
    operation: str,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    combined, values = _horizontal_value_plan(frames)
    if operation == "sum_frames":
        expression = sum(values[1:], values[0])
    elif operation == "mean":
        expression = sum(values[1:], values[0]) / len(values)
    elif operation == "product":
        expression = pl.fold(
            acc=pl.lit(1.0),
            function=lambda acc, value: acc * value,
            exprs=values,
        )
    elif operation == "minimum":
        expression = pl.min_horizontal(values)
    elif operation == "maximum":
        expression = pl.max_horizontal(values)
    else:
        raw_weights = config.get("weights")
        if not isinstance(raw_weights, Sequence):
            raise TypeError("weights must be a sequence")
        weights = tuple(raw_weights)
        if len(values) != len(weights):
            raise ValueError("weights length must match frame count")
        if any(
            not isinstance(weight, Real) or isinstance(weight, bool)
            for weight in weights
        ):
            raise TypeError("weights must be real numbers")
        expression = sum(
            (
                value * float(weight)
                for value, weight in zip(values, weights, strict=True)
            ),
            pl.lit(0.0),
        )
        if operation == "weighted_mean":
            total = float(sum(weights))
            if total == 0:
                raise ValueError("weights must not sum to zero")
            expression = expression / total
    return (
        combined.select(
            "time",
            "asset_id",
            expression.alias("value"),
        ),
        order,
        asset_time_ordered,
    )


for _plan_name, _plan_composer in {
    "sum_frames": sum_frames,
    "mean": mean,
    "product": product,
    "minimum": minimum,
    "maximum": maximum,
    "weighted_sum": weighted_sum,
    "weighted_mean": weighted_mean,
}.items():
    _plan_composer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frames, config, order, asset_time_ordered, name=_plan_name: (
            _plan_aggregation(
                frames,
                config,
                name,
                order,
                asset_time_ordered,
            )
        )
    )
