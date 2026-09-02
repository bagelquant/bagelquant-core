"""Execution contracts used by the graph planner.

The public decorators remain intentionally small.  Every registered operation
receives a contract so the runtime can safely keep Polars-native work lazy and
insert dense or eager barriers only when the operation's semantics require it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

import polars as pl


class ExecutionMode(StrEnum):
    LAZY = "lazy"
    EAGER_BARRIER = "eager_barrier"


class InputDensity(StrEnum):
    SPARSE_OK = "sparse_ok"
    DENSE_REQUIRED = "dense_required"


class TraceRule(StrEnum):
    NONE = "none"
    PASSTHROUGH = "passthrough"
    PARENT_MAX = "parent_max"
    SHIFT = "shift"
    CURRENT_AND_SHIFT_MAX = "current_and_shift_max"
    ROLLING_MAX = "rolling_max"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    CUSTOM = "custom"


TraceFunction = Callable[
    [tuple[pl.LazyFrame, ...], pl.LazyFrame, dict[str, Any], tuple[str, ...]],
    pl.LazyFrame,
]


@dataclass(frozen=True, slots=True)
class OperationContract:
    execution: ExecutionMode = ExecutionMode.LAZY
    density: InputDensity = InputDensity.SPARSE_OK
    trace_rule: TraceRule = TraceRule.PASSTHROUGH
    deterministic: bool = True
    trace_function: TraceFunction | None = None

    def __post_init__(self) -> None:
        if self.trace_rule == TraceRule.CUSTOM and self.trace_function is None:
            raise ValueError("custom trace rules require trace_function")
        if self.trace_rule != TraceRule.CUSTOM and self.trace_function is not None:
            raise ValueError("trace_function is only valid for custom trace rules")


_DENSE_TRANSFORMERS = {
    "bfill",
    "constant",
    "date_age_constraint",
    "diff",
    "diff_from_last_change",
    "ffill",
    "fillna",
    "fillna_zero",
    "lag",
    "pct_change",
    "pct_change_from_last_change",
    "remove_repeated",
}
_EAGER_TRANSFORMERS = {
    "orthogonalize",
    "rolling_elastic_net",
    "rolling_lasso",
    "rolling_ols",
    "rolling_percentile",
    "rolling_rank",
    "rolling_ridge",
}
_EAGER_COMPOSERS = {"broadcast_by_time"}

_PANEL_PARAMETER_TRANSFORMERS = {
    "group_demean",
    "group_max",
    "group_mean",
    "group_median",
    "group_min",
    "group_percentile",
    "group_rank",
    "group_rankpct",
    "group_std",
    "group_zscore",
    "mask",
    "orthogonalize",
    "project",
    "rolling_elastic_net",
    "rolling_lasso",
    "rolling_ols",
    "rolling_ridge",
    "vol_scale",
}


def default_operation_contract(
    operation: Callable[..., Any],
    *,
    kind: str,
) -> OperationContract:
    """Return a conservative contract for built-ins and external extensions."""

    module = getattr(operation, "__module__", "")
    name = getattr(operation, "__name__", "")
    builtin = module.startswith("bagelquant_core.")
    if not builtin:
        return OperationContract(
            execution=ExecutionMode.EAGER_BARRIER,
            density=InputDensity.DENSE_REQUIRED,
            trace_rule=TraceRule.NONE,
        )

    dense = (
        name in _DENSE_TRANSFORMERS
        or name.startswith("rolling_")
        or name.startswith("ewm_")
    )
    eager = (
        name in _EAGER_TRANSFORMERS
        if kind == "transformer"
        else name in _EAGER_COMPOSERS
    )
    trace_rule = (
        _transformer_trace_rule(name)
        if kind == "transformer"
        else TraceRule.PARENT_MAX
    )
    return OperationContract(
        execution=(
            ExecutionMode.EAGER_BARRIER if eager else ExecutionMode.LAZY
        ),
        density=(
            InputDensity.DENSE_REQUIRED if dense else InputDensity.SPARSE_OK
        ),
        trace_rule=trace_rule,
    )


def _transformer_trace_rule(name: str) -> TraceRule:
    if name in _PANEL_PARAMETER_TRANSFORMERS:
        return TraceRule.PARENT_MAX
    if name == "lag":
        return TraceRule.SHIFT
    if name in {
        "diff",
        "diff_from_last_change",
        "pct_change",
        "pct_change_from_last_change",
    }:
        return TraceRule.CURRENT_AND_SHIFT_MAX
    if name.startswith("rolling_") or name.startswith("ewm_"):
        return TraceRule.ROLLING_MAX
    if name == "ffill":
        return TraceRule.FORWARD_FILL
    if name == "bfill":
        return TraceRule.BACKWARD_FILL
    return TraceRule.PASSTHROUGH


__all__ = [
    "ExecutionMode",
    "InputDensity",
    "OperationContract",
    "TraceFunction",
    "TraceRule",
    "default_operation_contract",
]
