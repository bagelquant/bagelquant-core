"""General composers."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import polars as pl

from ..frame import VALUE, nary, panel_like
from ..operation_contract import ExecutionMode, OperationContract, TraceRule
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


def _broadcast_by_time_traces(
    frames: tuple[pl.LazyFrame, ...],
    result: pl.LazyFrame,
    _config: dict[str, object],
    traces: tuple[str, ...],
) -> pl.LazyFrame:
    source, like = frames
    output = result.select("time", "asset_id")
    expressions: list[pl.Expr] = []
    trace_names: dict[str, list[str]] = {trace: [] for trace in traces}
    source_columns = set(source.collect_schema().names())
    like_columns = set(like.collect_schema().names())

    source_trace_expressions: list[pl.Expr] = []
    for trace in traces:
        if trace in source_columns:
            name = f"__source_{trace}"
            trace_names[trace].append(name)
            source_trace_expressions.append(pl.col(trace).max().alias(name))
    if source_trace_expressions:
        output = output.join(
            source.group_by("time").agg(source_trace_expressions),
            on="time",
            how="left",
        )

    like_trace_expressions: list[pl.Expr] = []
    for trace in traces:
        if trace in like_columns:
            name = f"__like_{trace}"
            trace_names[trace].append(name)
            like_trace_expressions.append(pl.col(trace).alias(name))
    if like_trace_expressions:
        output = output.join(
            like.select("time", "asset_id", *like_trace_expressions),
            on=["time", "asset_id"],
            how="left",
        )

    for trace, names in trace_names.items():
        if names:
            expressions.append(pl.max_horizontal(*names).alias(trace))
    return output.with_columns(expressions).select(
        "time", "asset_id", *traces
    )


@composer(
    contract=OperationContract(
        execution=ExecutionMode.EAGER_BARRIER,
        trace_rule=TraceRule.CUSTOM,
        trace_function=_broadcast_by_time_traces,
    )
)
def broadcast_by_time(
    source: pl.DataFrame,
    like: pl.DataFrame,
) -> pl.DataFrame:
    """Broadcast one source value per date to the keyed rows of another Panel."""

    counts = source.group_by("time").len().filter(pl.col("len") != 1)
    if counts.height:
        dates = counts.get_column("time").head(5).to_list()
        raise ValueError(
            "broadcast_by_time requires exactly one source row per date; "
            f"invalid dates include {dates}"
        )
    return (
        like.select("time", "asset_id")
        .join(
            source.select("time", pl.col(VALUE).alias("__source_value")),
            on="time",
            how="inner",
        )
        .select("time", "asset_id", pl.col("__source_value").alias(VALUE))
        .sort("time", "asset_id")
    )


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
