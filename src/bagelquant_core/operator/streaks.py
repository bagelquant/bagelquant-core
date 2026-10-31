"""Per-asset run-length and directional-streak operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from bagelquant_core.frame import ASSET_ID, TIME, VALUE
from bagelquant_core.operation_contract import InputDensity, OperationContract, TraceRule
from bagelquant_core.operator._definition import operator
from bagelquant_core.streak_state import checkpoint_streak, valid_streak_state

_VALID = "__bq_streak_valid"
_PREVIOUS = "__bq_streak_previous"
_PREVIOUS_VALID = "__bq_streak_previous_valid"
_DIRECTION = "__bq_streak_direction"
_PREVIOUS_DIRECTION = "__bq_streak_previous_direction"
_EFFECTIVE_DIRECTION = "__bq_streak_effective_direction"
_BLOCK = "__bq_streak_block"
_RUN = "__bq_streak_run"
_RUN_START = "__bq_streak_run_start"
_REVERSAL = "__bq_streak_reversal"
_POSITION = "__bq_streak_position"
_OFFSET = "__bq_streak_offset"
_FIRST = "__bq_streak_first"
_SEED_VALUE = "__bq_streak_seed_value"
_SEED_DIRECTION = "__bq_streak_seed_direction"
_SEED_COUNT = "__bq_streak_seed_count"

Frame = pl.DataFrame | pl.LazyFrame


def _valid_value() -> pl.Expr:
    value = pl.col(VALUE)
    return value.is_not_null() & ~value.is_nan()


def _repeat_state(frame: Frame) -> Frame:
    value = pl.col(VALUE)
    prepared = frame.with_columns(
        _valid_value().alias(_VALID),
        value.shift(1).over(ASSET_ID).alias(_PREVIOUS),
    ).with_columns(
        (
            pl.col(_PREVIOUS).is_not_null()
            & ~pl.col(_PREVIOUS).is_nan()
        ).alias(_PREVIOUS_VALID)
    )
    prepared = prepared.with_columns(
        (
            pl.col(_VALID)
            & pl.col(_PREVIOUS_VALID)
            & (value == pl.col(_PREVIOUS))
        ).alias("__bq_streak_same")
    ).with_columns(
        (
            pl.col(_VALID) & ~pl.col("__bq_streak_same")
        ).alias(_RUN_START)
    )
    prepared = prepared.with_columns(
        pl.col(_RUN_START)
        .cast(pl.Int64)
        .cum_sum()
        .over(ASSET_ID)
        .alias(_RUN)
    )
    return prepared.with_columns(
        pl.when(pl.col(_VALID))
        .then(
            pl.col(_VALID)
            .cast(pl.Int64)
            .cum_sum()
            .over([ASSET_ID, _RUN])
        )
        .otherwise(None)
        .cast(pl.Int64)
        .alias(VALUE)
    )


def _change_state(frame: Frame, *, restored: bool = False) -> Frame:
    value = pl.col(VALUE)
    previous = value.shift(1).over(ASSET_ID)
    if restored:
        previous = pl.when(pl.col(_FIRST)).then(pl.col(_SEED_VALUE)).otherwise(previous)
    prepared = frame.with_columns(
        _valid_value().alias(_VALID),
        previous.alias(_PREVIOUS),
    ).with_columns(
        (
            pl.col(_PREVIOUS).is_not_null()
            & ~pl.col(_PREVIOUS).is_nan()
        ).alias(_PREVIOUS_VALID)
    )
    return prepared


def _streak_state(frame: Frame, *, reset_on_equal: bool, restored: bool = False) -> Frame:
    prepared = _change_state(frame, restored=restored).with_columns(
        pl.when(pl.col(_VALID) & pl.col(_PREVIOUS_VALID))
        .then((pl.col(VALUE) - pl.col(_PREVIOUS)).sign())
        .otherwise(None)
        .cast(pl.Int8)
        .alias(_DIRECTION)
    )
    if reset_on_equal:
        return _resetting_streak_state(prepared, restored=restored)
    return _holding_streak_state(prepared, restored=restored)


def _resetting_streak_state(frame: Frame, *, restored: bool = False) -> Frame:
    previous = pl.col(_DIRECTION).shift(1).over(ASSET_ID)
    if restored:
        previous = pl.when(pl.col(_FIRST)).then(pl.col(_SEED_DIRECTION)).otherwise(previous)
    prepared = frame.with_columns(
        previous.alias(_PREVIOUS_DIRECTION)
    ).with_columns(
        (
            pl.col(_VALID)
            & (
                pl.col(_DIRECTION).is_null()
                | (pl.col(_DIRECTION) == 0)
                | pl.col(_PREVIOUS_DIRECTION).is_null()
                | (pl.col(_DIRECTION) != pl.col(_PREVIOUS_DIRECTION))
            )
        ).alias(_RUN_START),
        (
            pl.col(_VALID)
            & pl.col(_DIRECTION).is_not_null()
            & pl.col(_PREVIOUS_DIRECTION).is_not_null()
            & (pl.col(_DIRECTION) != 0)
            & (pl.col(_PREVIOUS_DIRECTION) != 0)
            & (pl.col(_DIRECTION) != pl.col(_PREVIOUS_DIRECTION))
        ).alias(_REVERSAL),
    )
    return _finish_streak_state(prepared, hold_equal=False, restored=restored)


def _holding_streak_state(frame: Frame, *, restored: bool = False) -> Frame:
    prepared = frame.with_columns(
        (pl.col(_VALID) & ~pl.col(_PREVIOUS_VALID)).alias(_RUN_START)
    ).with_columns(
        pl.col(_RUN_START)
        .cast(pl.Int64)
        .cum_sum()
        .over(ASSET_ID)
        .alias(_BLOCK)
    )
    effective = (
        pl.when(
            pl.col(_DIRECTION).is_not_null()
            & (pl.col(_DIRECTION) != 0)
        )
        .then(pl.col(_DIRECTION))
        .otherwise(None)
        .forward_fill()
        .over([ASSET_ID, _BLOCK])
    )
    if restored:
        effective = pl.when(pl.col(_BLOCK) == 0).then(
            pl.coalesce(effective, pl.col(_SEED_DIRECTION))).otherwise(effective)
    previous = pl.col(_EFFECTIVE_DIRECTION).shift(1).over(ASSET_ID)
    if restored:
        previous = pl.when(pl.col(_FIRST)).then(pl.col(_SEED_DIRECTION)).otherwise(previous)
    prepared = prepared.with_columns(effective.alias(_EFFECTIVE_DIRECTION)).with_columns(
        previous.alias(_PREVIOUS_DIRECTION))
    prepared = prepared.with_columns(
        (
            pl.col(_VALID)
            & pl.col(_PREVIOUS_VALID)
            & pl.col(_DIRECTION).is_not_null()
            & (pl.col(_DIRECTION) != 0)
            & pl.col(_PREVIOUS_DIRECTION).is_not_null()
            & (pl.col(_DIRECTION) != pl.col(_PREVIOUS_DIRECTION))
        ).alias(_REVERSAL)
    ).with_columns(
        (pl.col(_RUN_START) | pl.col(_REVERSAL)).alias(_RUN_START)
    )
    return _finish_streak_state(prepared, hold_equal=True, restored=restored)


def _finish_streak_state(frame: Frame, *, hold_equal: bool, restored: bool = False) -> Frame:
    prepared = frame.with_columns(
        pl.col(_RUN_START)
        .cast(pl.Int64)
        .cum_sum()
        .over(ASSET_ID)
        .alias(_RUN)
    )
    movement = (
        pl.col(_DIRECTION).is_not_null() & (pl.col(_DIRECTION) != 0)
    )
    prepared = prepared.with_columns(
        (
            movement.cast(pl.Int64)
            if hold_equal
            else pl.col(_VALID).cast(pl.Int64)
        )
        .cum_sum()
        .over([ASSET_ID, _RUN])
        .alias(_POSITION),
        pl.col(_REVERSAL)
        .cast(pl.Int64)
        .max()
        .over([ASSET_ID, _RUN])
        .alias(_OFFSET),
    )
    direction = (
        pl.col(_EFFECTIVE_DIRECTION)
        if hold_equal
        else pl.col(_DIRECTION)
    )
    carry = (pl.when(pl.col(_RUN) == 0).then(pl.col(_SEED_COUNT).fill_null(0)).otherwise(0)
             if restored else pl.lit(0))
    return prepared.with_columns(
        pl.when(~pl.col(_VALID))
        .then(None)
        .when(direction.is_null() | ((direction == 0) & ~pl.lit(hold_equal)))
        .then(0)
        .otherwise(
            direction.cast(pl.Int64)
            * (pl.col(_POSITION) - pl.col(_OFFSET))
            + carry
        )
        .cast(pl.Int64)
        .alias(VALUE)
    )


def _run_trace(
    frame: Frame,
    *,
    traces: tuple[str, ...],
    restored: bool = False,
) -> Frame:
    dependency_columns = {
        trace: f"__bq_streak_trace_dependency_{index}"
        for index, trace in enumerate(traces)
    }
    dependencies = []
    for index, (trace, dependency) in enumerate(dependency_columns.items()):
        previous = pl.col(trace).cast(pl.Int32).shift(1).over(ASSET_ID)
        if restored:
            previous = pl.when(pl.col(_FIRST)).then(
                pl.col(f"__bq_streak_seed_previous_trace_{index}").cast(pl.Int32)).otherwise(previous)
        dependencies.append(pl.max_horizontal(pl.col(trace).cast(pl.Int32), previous).alias(dependency))
    prepared = frame.with_columns(dependencies)
    expressions: list[pl.Expr] = []
    for index, trace in enumerate(traces):
        cumulative = (
            pl.col(dependency_columns[trace])
            .cum_max()
            .over([ASSET_ID, _RUN])
            .cast(pl.Date)
        )
        if restored:
            cumulative = pl.when(pl.col(_RUN) == 0).then(pl.max_horizontal(
                cumulative, pl.col(f"__bq_streak_seed_run_trace_{index}"))).otherwise(cumulative)
        expressions.append(
            pl.when(pl.col(_VALID))
            .then(cumulative)
            .otherwise(pl.col(trace))
            .alias(trace)
        )
    return prepared.with_columns(expressions).select(TIME, ASSET_ID, *traces)


def _repeat_trace(
    parents: tuple[pl.LazyFrame, ...],
    _result: pl.LazyFrame,
    _config: dict[str, Any],
    traces: tuple[str, ...],
) -> pl.LazyFrame:
    state = _repeat_state(parents[0].sort([ASSET_ID, TIME]))
    traced = _run_trace(state, traces=traces)
    assert isinstance(traced, pl.LazyFrame)
    return traced


def _streak_trace(
    parents: tuple[pl.LazyFrame, ...],
    _result: pl.LazyFrame,
    config: dict[str, Any],
    traces: tuple[str, ...],
) -> pl.LazyFrame:
    reset_on_equal = config.get("reset_on_equal", True)
    _validate_reset_on_equal(reset_on_equal)
    from bagelquant_core.operator_state import checkpoint_capture_enabled
    if checkpoint_capture_enabled():
        traced = checkpoint_streak(parents[0].collect(), reset_on_equal=reset_on_equal, traces=traces)
        return traced.select(TIME, ASSET_ID, *traces).lazy()
    state = _streak_state(
        parents[0].sort([ASSET_ID, TIME]),
        reset_on_equal=reset_on_equal,
    )
    traced = _run_trace(state, traces=traces)
    assert isinstance(traced, pl.LazyFrame)
    return traced


_REPEAT_CONTRACT = OperationContract(
    density=InputDensity.DENSE_REQUIRED,
    trace_rule=TraceRule.CUSTOM,
    trace_function=_repeat_trace,
)
_STREAK_CONTRACT = OperationContract(
    density=InputDensity.DENSE_REQUIRED,
    trace_rule=TraceRule.CUSTOM,
    trace_function=_streak_trace,
    checkpoint_replay=True,
    checkpoint_history=0,
    checkpoint_validator=valid_streak_state,
)


@operator(contract=_REPEAT_CONTRACT)
def repeat_count(frame: pl.DataFrame) -> pl.DataFrame:
    state = _repeat_state(frame.sort([ASSET_ID, TIME]))
    assert isinstance(state, pl.DataFrame)
    return state.select(TIME, ASSET_ID, VALUE)


@operator
def diff_from_last_change(frame: pl.DataFrame) -> pl.DataFrame:
    state = _change_state(frame.sort([ASSET_ID, TIME]))
    assert isinstance(state, pl.DataFrame)
    return state.with_columns(
        pl.when(
            pl.col(_VALID)
            & pl.col(_PREVIOUS_VALID)
            & (pl.col(VALUE) != pl.col(_PREVIOUS))
        )
        .then(pl.col(VALUE) - pl.col(_PREVIOUS))
        .otherwise(None)
        .alias(VALUE)
    ).select(TIME, ASSET_ID, VALUE)


@operator
def pct_change_from_last_change(frame: pl.DataFrame) -> pl.DataFrame:
    state = _change_state(frame.sort([ASSET_ID, TIME]))
    assert isinstance(state, pl.DataFrame)
    return state.with_columns(
        pl.when(
            pl.col(_VALID)
            & pl.col(_PREVIOUS_VALID)
            & (pl.col(VALUE) != pl.col(_PREVIOUS))
        )
        .then(pl.col(VALUE) / pl.col(_PREVIOUS) - 1.0)
        .otherwise(None)
        .alias(VALUE)
    ).select(TIME, ASSET_ID, VALUE)


@operator(contract=_STREAK_CONTRACT)
def streak_count(
    frame: pl.DataFrame,
    *,
    reset_on_equal: bool = True,
) -> pl.DataFrame:
    _validate_reset_on_equal(reset_on_equal)
    continued = checkpoint_streak(frame, reset_on_equal=reset_on_equal)
    if continued is not None:
        return continued.select(TIME, ASSET_ID, VALUE)
    state = _streak_state(
        frame.sort([ASSET_ID, TIME]),
        reset_on_equal=reset_on_equal,
    )
    assert isinstance(state, pl.DataFrame)
    return state.select(TIME, ASSET_ID, VALUE)


def _validate_reset_on_equal(value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError("streak_count reset_on_equal must be a boolean")


def _ordered_plan(
    frame: pl.LazyFrame,
    config: Mapping[str, Any],
    order: str | None,
    asset_time_ordered: bool,
    *,
    operation: str,
) -> tuple[pl.LazyFrame, str | None, bool]:
    source = frame if asset_time_ordered else frame.sort([ASSET_ID, TIME])
    output_order = order if asset_time_ordered else "asset_time"
    if operation == "repeat_count":
        state = _repeat_state(source)
    elif operation == "streak_count":
        reset_on_equal = config.get("reset_on_equal", True)
        _validate_reset_on_equal(reset_on_equal)
        state = _streak_state(source, reset_on_equal=reset_on_equal)
    else:
        state = _change_state(source)
        changed = (
            pl.col(_VALID)
            & pl.col(_PREVIOUS_VALID)
            & (pl.col(VALUE) != pl.col(_PREVIOUS))
        )
        expression = (
            pl.col(VALUE) - pl.col(_PREVIOUS)
            if operation == "diff_from_last_change"
            else pl.col(VALUE) / pl.col(_PREVIOUS) - 1.0
        )
        state = state.with_columns(
            pl.when(changed)
            .then(expression)
            .otherwise(None)
            .alias(VALUE)
        )
    assert isinstance(state, pl.LazyFrame)
    return (
        state.select(TIME, ASSET_ID, VALUE),
        output_order,
        True,
    )


for _plan_name, _plan_operator in {
    "repeat_count": repeat_count,
    "diff_from_last_change": diff_from_last_change,
    "pct_change_from_last_change": pct_change_from_last_change,
    "streak_count": streak_count,
}.items():
    _plan_operator._set_plan_operation(  # type: ignore[attr-defined]
        lambda frame, config, order, asset_time_ordered, name=_plan_name: (
            _ordered_plan(
                frame,
                config,
                order,
                asset_time_ordered,
                operation=name,
            )
        )
    )


__all__ = [
    "diff_from_last_change",
    "pct_change_from_last_change",
    "repeat_count",
    "streak_count",
]
