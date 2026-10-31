"""Lazy dependency-date propagation across daily and grouped calculations."""
from __future__ import annotations

import polars as pl

from .frame import ASSET_ID, TIME, VALUE
from .operation_contract import TraceRule


def _trace_dtypes(parents, traces):
    """Choose lossless channel types from schema metadata, never value samples."""
    schemas = [parent.trace_frame.collect_schema() for parent in parents if parent.trace_frame is not None]
    result = {}
    precision = {"ms": 0, "us": 1, "ns": 2}
    for trace in traces:
        declared = [schema[trace] for schema in schemas if trace in schema and schema[trace] != pl.Null]
        if not declared:
            result[trace] = pl.Null
        elif all(dtype == declared[0] for dtype in declared):
            result[trace] = declared[0]
        elif all(dtype == pl.Date or isinstance(dtype, pl.Datetime) for dtype in declared):
            datetimes = [dtype for dtype in declared if isinstance(dtype, pl.Datetime)]
            if len({dtype.time_zone for dtype in datetimes}) != 1:
                raise ValueError(f"trace channel {trace!r} requires matching time zones")
            chosen = max(datetimes, key=lambda dtype: precision[dtype.time_unit])
            result[trace] = pl.Datetime(chosen.time_unit, chosen.time_zone)
        else:
            raise TypeError(f"trace channel {trace!r} has incompatible dependency dtypes")
    return result


def cross_sectional_traces(keys, parents, rule, config, traces):
    """Bind each output to all consumed peer dates without reading other periods.

    Group values are scoped to the selected group. Classification dates cover
    the complete classification cross-section: deciding that a peer is outside
    the group consumes its assignment as well as assignments inside the group.
    """
    dtypes = _trace_dtypes(parents, traces)
    typed = [None if parent.trace_frame is None else parent.trace_frame.with_columns(
        pl.col(trace).cast(dtypes[trace]) for trace in parent.trace_columns) for parent in parents]
    if rule == TraceRule.GROUP_MAX:
        source, groups = parents
        labels = groups.frame.select(TIME, ASSET_ID, pl.col(VALUE).alias("__group"))
        source_dates = typed[0]
        if source_dates is None:
            source_dates = source.frame.select(TIME, ASSET_ID)
        joined = source_dates.join(labels, on=[TIME, ASSET_ID], how="inner").drop_nulls("__group")
        available = [trace for trace in traces if trace in source.trace_columns]
        grouped = joined.group_by(TIME, "__group").agg(pl.col(trace).max() for trace in available)
        output = keys.join(labels, on=[TIME, ASSET_ID], how="left", maintain_order="left").join(
            grouped, on=[TIME, "__group"], how="left", maintain_order="left").drop("__group")
        for trace in traces:
            if trace in groups.trace_columns:
                label = f"__classification_{trace}"
                daily = typed[1].group_by(TIME).agg(pl.col(trace).max().alias(label))
                output = output.join(daily, on=TIME, how="left", maintain_order="left")
                output = output.with_columns((pl.max_horizontal(trace, label) if trace in available
                    else pl.col(label)).alias(trace)).drop(label)
        return output.select(TIME, ASSET_ID, *traces)

    parts = [frame.select(TIME,
        *[pl.col(trace) if trace in parent.trace_columns else pl.lit(None, dtype=dtypes[trace]).alias(trace)
          for trace in traces]) for parent, frame in zip(parents, typed, strict=True) if frame is not None]
    daily = pl.concat(parts).group_by(TIME).agg(pl.col(trace).max() for trace in traces).sort(TIME)
    if rule == TraceRule.CROSS_SECTION_CUMULATIVE_MAX:
        daily = daily.with_columns((pl.col(trace) if dtypes[trace] == pl.Null else
            pl.col(trace).cum_max().forward_fill()).alias(trace) for trace in traces)
    elif rule == TraceRule.CROSS_SECTION_ROLLING_MAX:
        # Cross-sectional preparation precedes each asset's numerical rolling
        # window. Dynamic membership may omit different dates per asset; a
        # daily rolling table would substitute other dates for sparse reentry.
        scoped = parents[0].frame.select(TIME, ASSET_ID).join(daily, on=TIME,
            how="left", maintain_order="left").sort(ASSET_ID, TIME)
        schema = scoped.collect_schema()
        traced = scoped.with_columns(pl.col(trace).cast(pl.Int64)
            .rolling_max(window_size=config["window"], min_samples=1).over(ASSET_ID)
            .cast(schema[trace]).alias(trace) for trace in traces)
        return keys.join(traced, on=[TIME, ASSET_ID], how="left", maintain_order="left")
    return keys.join(daily, on=TIME, how="left", maintain_order="left")
