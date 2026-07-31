"""Scaling composers."""

from __future__ import annotations

import polars as pl

from ..frame import binary
from .core import _horizontal_value_plan, composer


@composer
def vol_scale(frame: pl.DataFrame, volatility: pl.DataFrame) -> pl.DataFrame:
    return binary(frame, volatility, lambda value, vol: value / vol)


def _plan_vol_scale(
    frames: tuple[pl.LazyFrame, ...],
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    combined, values = _horizontal_value_plan(frames)
    return (
        combined.select(
            "time",
            "asset_id",
            (values[0] / values[1]).alias("value"),
        ),
        order,
        asset_time_ordered,
    )


vol_scale._set_plan_operation(  # type: ignore[attr-defined]
    lambda frames, config, order, asset_time_ordered: _plan_vol_scale(
        frames,
        order,
        asset_time_ordered,
    )
)
