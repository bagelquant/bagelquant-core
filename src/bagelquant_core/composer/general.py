"""General composers."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import polars as pl

from ..frame import VALUE, nary, panel_like
from ..transformer.core import transformer
from .core import _horizontal_value_plan, composer


@transformer
def project(frame: pl.DataFrame, *, binary: pl.DataFrame) -> pl.DataFrame:
    data = frame.rename({VALUE: "x"}).join(
        binary.rename({VALUE: "binary"}),
        on=["time", "asset_id"],
        how="inner",
    )
    return panel_like(
        data,
        pl.when(pl.col("binary") == 1.0).then(pl.col("x")).otherwise(None),
    )


@transformer
def mask(
    frame: pl.DataFrame,
    *,
    mask_frame: pl.DataFrame,
    replace_value: float = float("nan"),
) -> pl.DataFrame:
    if not isinstance(replace_value, Real) or isinstance(replace_value, bool):
        raise TypeError("mask replace_value must be a real number")
    data = frame.rename({VALUE: "x"}).join(
        mask_frame.rename({VALUE: "mask"}),
        on=["time", "asset_id"],
        how="inner",
    )
    condition = pl.col("mask").fill_nan(None).fill_null(0.0).cast(pl.Boolean)
    return panel_like(
        data,
        pl.when(condition)
        .then(pl.col("x"))
        .otherwise(pl.lit(float(replace_value))),
    )


@composer
def coalesce(*frames: pl.DataFrame) -> pl.DataFrame:
    return nary(frames, lambda values: pl.coalesce(values))


def _plan_general(
    frames: tuple[pl.LazyFrame, ...],
    config: Mapping[str, object],
    operation: str,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    combined, values = _horizontal_value_plan(frames)
    if operation == "project":
        expression = (
            pl.when(values[1] == 1.0).then(values[0]).otherwise(None)
        )
    elif operation == "mask":
        replace_value = config.get("replace_value", float("nan"))
        if (
            not isinstance(replace_value, Real)
            or isinstance(replace_value, bool)
        ):
            raise TypeError("mask replace_value must be a real number")
        condition = values[1].fill_nan(None).fill_null(0.0).cast(
            pl.Boolean
        )
        expression = (
            pl.when(condition)
            .then(values[0])
            .otherwise(pl.lit(float(replace_value)))
        )
    else:
        expression = pl.coalesce(values)
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
    "coalesce": coalesce,
}.items():
    _plan_composer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frames, config, order, asset_time_ordered, name=_plan_name: (
            _plan_general(
                frames,
                config,
                name,
                order,
                asset_time_ordered,
            )
        )
    )
