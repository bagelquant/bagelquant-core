"""Vectorized directional streak continuation over admitted input coordinates."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import math

import polars as pl

from .operator_state import (
    checkpoint_capture_enabled, operator_execution_calendar, restored_operator_state, save_operator_state,
)

STATE_SCHEMA = "streak_state.v1"
_INPUT = "__bq_streak_input"


def validate_streak_calendar(domain):
    """Reject unproven checkpoint coordinates using Domain metadata only.

    Stored-parent prefix indexes and zero-history continuation currently use
    the explicit execution calendar. Dynamic off-calendar keys therefore need
    a separate proven coordinate contract before they can admit saved state.
    """
    if domain.is_dynamic:
        outside = domain.grid_lazy().select(pl.col("time").filter(
            ~pl.col("time").is_in(domain.times.implode())).min()).collect().item()
        if outside is not None:
            raise ValueError(f"Streak checkpoint active date {outside} is outside the execution calendar; "
                "provide a complete anchored calendar or explicit coordinate-contract maintenance before updating")


def _date(value):
    return value is None or isinstance(value, str) and date.fromisoformat(value).isoformat() == value


def _number(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else "inf" if value > 0 else "-inf"
    return value


def valid_streak_state(state):
    """Validate retained values, directions, counts and every custom trace channel."""
    if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
        return False
    try:
        if not isinstance(state.get("through"), str) or not _date(state["through"]):
            return False
        traces, assets = state.get("traces"), state.get("assets")
        if (not isinstance(traces, list) or any(not isinstance(name, str) for name in traces)
                or len(set(traces)) != len(traces) or not isinstance(assets, Mapping)):
            return False
        for asset, value in assets.items():
            if not isinstance(asset, str) or not isinstance(value, Mapping):
                return False
            if set(value) != {"value", "direction", "count", "previous_traces", "run_traces"}:
                return False
            previous, direction, count = (value.get(name) for name in ("value", "direction", "count"))
            if previous is not None and not (
                    type(previous) in (float, int) and math.isfinite(previous)
                    or isinstance(previous, str) and previous in {"inf", "-inf"}):
                return False
            if direction is not None and (type(direction) is not int or direction not in {-1, 0, 1}):
                return False
            if count is not None and (type(count) is not int or not -2**63 <= count < 2**63):
                return False
            if (previous is None) != (count is None):
                return False
            if count and (direction not in {-1, 1} or (count > 0) != (direction > 0)):
                return False
            for channel in ("previous_traces", "run_traces"):
                entries = value.get(channel)
                if not isinstance(entries, Mapping) or set(entries) != set(traces) or not all(
                        _date(day) for day in entries.values()):
                    return False
    except (ValueError, TypeError, OverflowError):
        return False
    return True


def checkpoint_streak(frame, *, reset_on_equal, traces=()):
    """Reuse the ordinary vectorized formula, adding only proven boundary carries.

    Only per-asset checkpoint exchange uses Python objects. Every row transition
    and run reduction remains in the original Polars state/trace implementation.
    The explicit execution calendar advances empty sessions without inventing rows.
    """
    if not checkpoint_capture_enabled():
        return None
    from .operator.streaks import (
        _DIRECTION, _EFFECTIVE_DIRECTION, _FIRST, _SEED_VALUE, _SEED_DIRECTION, _SEED_COUNT,
        _streak_state, _run_trace,
    )
    previous = restored_operator_state()
    if previous is not None and not valid_streak_state(previous):
        raise ValueError("Streak checkpoint state is invalid; explicit checkpoint maintenance is required")
    if traces and previous is not None and previous["traces"] != list(traces):
        raise ValueError("Streak checkpoint trace channels changed")
    ordered = frame.sort("asset_id", "time")
    if previous is not None:
        ordered = ordered.filter(pl.col("time") > date.fromisoformat(previous["through"]))
    ordered = ordered.with_columns(pl.col("value").alias(_INPUT))
    states = {} if previous is None else {asset: dict(value) for asset, value in previous["assets"].items()}
    if previous is not None:
        rows = []
        schema = {"asset_id": pl.String, _SEED_VALUE: frame.schema["value"],
            _SEED_DIRECTION: pl.Int8, _SEED_COUNT: pl.Int64}
        for index in range(len(traces)):
            schema[f"__bq_streak_seed_previous_trace_{index}"] = pl.Date
            schema[f"__bq_streak_seed_run_trace_{index}"] = pl.Date
        for asset, value in states.items():
            raw = value["value"]
            row = {"asset_id": asset, _SEED_VALUE: float(raw) if isinstance(raw, str) else raw,
                _SEED_DIRECTION: value["direction"], _SEED_COUNT: value["count"]}
            for index, trace in enumerate(traces):
                row[f"__bq_streak_seed_previous_trace_{index}"] = (
                    None if value["previous_traces"][trace] is None else date.fromisoformat(value["previous_traces"][trace]))
                row[f"__bq_streak_seed_run_trace_{index}"] = (
                    None if value["run_traces"][trace] is None else date.fromisoformat(value["run_traces"][trace]))
            rows.append(row)
        seeds = pl.DataFrame(rows, schema=schema)
        ordered = ordered.join(seeds, on="asset_id", how="left", maintain_order="left").with_columns(
            (pl.int_range(pl.len()).over("asset_id") == 0).alias(_FIRST))
    state = _streak_state(ordered, reset_on_equal=reset_on_equal, restored=previous is not None)
    result = state.select("time", "asset_id", "value")
    if traces:
        traced = _run_trace(state, traces=traces, restored=previous is not None)
        result = result.join(traced, on=["time", "asset_id"], how="left", maintain_order="left")
    direction = _DIRECTION if reset_on_equal else _EFFECTIVE_DIRECTION
    tail = state.group_by("asset_id", maintain_order=True).tail(1).select(
        "time", "asset_id", _INPUT, direction, pl.col("value").alias("count"), *traces)
    if traces:
        tail = tail.join(result.select("time", "asset_id", *[
            pl.col(trace).alias(f"__bq_streak_final_trace_{index}") for index, trace in enumerate(traces)]),
            on=["time", "asset_id"], how="left")
    for row in tail.iter_rows(named=True):
        states[row["asset_id"]] = {"value": _number(row[_INPUT]), "direction": row[direction], "count": row["count"],
            "previous_traces": {trace: None if row[trace] is None else str(row[trace]) for trace in traces},
            "run_traces": {trace: None if row[f"__bq_streak_final_trace_{index}"] is None else
                str(row[f"__bq_streak_final_trace_{index}"]) for index, trace in enumerate(traces)}}
    # Numeric evaluation precedes the custom trace pass. Preserve unobserved
    # assets and their trace carry until that pass supplies the current channels.
    if not traces:
        states = {asset: {**value, "previous_traces": {}, "run_traces": {}} for asset, value in states.items()}
    calendar = operator_execution_calendar()
    through = max(calendar) if calendar else frame["time"].max()
    if through is not None:
        saved = {"schema": STATE_SCHEMA, "through": str(through), "traces": list(traces), "assets": states}
        if not valid_streak_state(saved):
            raise ValueError("Streak execution produced invalid checkpoint state")
        save_operator_state(saved)
    return result
