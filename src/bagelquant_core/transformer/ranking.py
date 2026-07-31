"""Ranking transforms."""

from __future__ import annotations

import polars as pl

from ..frame import TIME, VALUE, cross_section_rank, panel_like, unary
from .core import _expression_plan, transformer


@transformer
def rankpct(frame: pl.DataFrame) -> pl.DataFrame:
    dense = pl.col(VALUE).rank("dense").over(TIME)
    distinct = pl.col(VALUE).n_unique().over(TIME)
    return panel_like(frame, dense / distinct)


@transformer
def nrank(frame: pl.DataFrame) -> pl.DataFrame:
    pct = cross_section_rank(frame, pct=True)
    return unary(pct, 2.0 * pl.col(VALUE) - 1.0)


@transformer
def logrank(frame: pl.DataFrame) -> pl.DataFrame:
    ranked = cross_section_rank(frame, pct=True)
    return unary(ranked, pl.col(VALUE).log())


def _plan_ranking(
    frame: pl.LazyFrame,
    operation: str,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    value = pl.col(VALUE)
    if operation == "rankpct":
        expression = (
            value.rank("dense").over(TIME)
            / value.n_unique().over(TIME)
        )
    else:
        percentile = (
            value.rank("average").over(TIME) / value.count().over(TIME)
        )
        expression = (
            2.0 * percentile - 1.0
            if operation == "nrank"
            else percentile.log()
        )
    return _expression_plan(
        frame,
        expression,
        order,
        asset_time_ordered,
    )


for _plan_name, _plan_transformer in {
    "rankpct": rankpct,
    "nrank": nrank,
    "logrank": logrank,
}.items():
    _plan_transformer._set_plan_operation(  # type: ignore[attr-defined]
        lambda frame, config, order, asset_time_ordered, name=_plan_name: (
            _plan_ranking(
                frame,
                name,
                order,
                asset_time_ordered,
            )
        )
    )
