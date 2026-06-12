"""Aggregation composers."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ..frame import VALUE, nary, panel_like
from .core import composer


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
    if len(frames) != len(weights):
        raise ValueError("weights length must match frame count")
    joined = frames[0].rename({VALUE: "v0"})
    expr = pl.col("v0") * float(weights[0])
    for index, (frame, weight) in enumerate(
        zip(frames[1:], weights[1:], strict=True), start=1
    ):
        name = f"v{index}"
        joined = joined.join(
            frame.rename({VALUE: name}), on=["time", "asset_id"], how="inner"
        )
        expr = expr + pl.col(name) * float(weight)
    return panel_like(joined, expr)


@composer
def weighted_mean(*frames: pl.DataFrame, weights: Sequence[float]) -> pl.DataFrame:
    total = float(sum(weights))
    if total == 0:
        raise ValueError("weights must not sum to zero")
    return panel_like(
        weighted_sum.operation(*frames, weights=weights), pl.col(VALUE) / total
    )


min = minimum
max = maximum
