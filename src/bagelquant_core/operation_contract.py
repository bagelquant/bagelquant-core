"""Execution contracts used by the graph planner.

The public decorators remain intentionally small.  Every registered operation
receives a contract so the runtime can safely keep Polars-native work lazy and
insert dense or eager barriers only when the operation's semantics require it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from inspect import signature, Parameter
from collections.abc import Mapping
from typing import Any, Callable, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from bagelquant_core.logical import LogicalGraphSpec


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
    CUMULATIVE_MAX = "cumulative_max"
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
    history: int | Callable[[Mapping[str, Any]], int | None] | None = None
    checkpoint_replay: bool = False
    trace_function: TraceFunction | None = None

    def __post_init__(self) -> None:
        if self.trace_rule == TraceRule.CUSTOM and self.trace_function is None:
            raise ValueError("custom trace rules require trace_function")
        if self.trace_rule != TraceRule.CUSTOM and self.trace_function is not None:
            raise ValueError("trace_function is only valid for custom trace rules")


_DENSE_OPERATIONS = {
    "bfill",
    "constant",
    "date_age_constraint",
    "diff",
    "diff_from_last_change",
    "ffill",
    "fillna",
    "fillna_zero",
    "lag",
    "kelly",
    "kelly_nonan_standardize",
    "kelly_rank_boxcox",
    "kelly_rescaling_weight",
    "pct_change",
    "pct_change_from_last_change",
    "remove_repeated",
}
_EAGER_REGRESSIONS = {
    "orthogonalize",
    "rolling_elastic_net",
    "rolling_lasso",
    "rolling_ols",
    "rolling_percentile",
    "rolling_rank",
    "rolling_ridge",
}
_EAGER_ALIGNMENT = {"broadcast_by_time"}

_AUXILIARY_OPERATIONS = {
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
    "project_domain",
}


def default_operation_contract(
    operation: Callable[..., Any],
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
        name in _DENSE_OPERATIONS
        or name.startswith("rolling_")
        or name.startswith("ewm_")
    )
    eager = name in _EAGER_REGRESSIONS | _EAGER_ALIGNMENT
    arity = tuple(p for p in signature(operation).parameters.values() if p.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD, Parameter.VAR_POSITIONAL})
    trace_rule = (
        _operator_trace_rule(name)
        if len(arity) == 1 and arity[0].kind != Parameter.VAR_POSITIONAL
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
        history=lambda parameters: _finite_history(operation, parameters),
        checkpoint_replay=name in {"ewm_mean", "ewm_var", "ewm_std"},
    )


def _operator_trace_rule(name: str) -> TraceRule:
    if name in _AUXILIARY_OPERATIONS:
        return TraceRule.PARENT_MAX
    if name == "lag":
        return TraceRule.SHIFT
    if name in {
        "diff",
        "diff_from_last_change",
        "pct_change",
        "pct_change_from_last_change",
        "remove_repeated",
    }:
        return TraceRule.CURRENT_AND_SHIFT_MAX
    if name.startswith("ewm_"):
        return TraceRule.CUMULATIVE_MAX
    if (
        name.startswith("rolling_")
        or name.startswith("kelly")
        or name == "date_age_constraint"
    ):
        return TraceRule.ROLLING_MAX
    if name == "ffill":
        return TraceRule.FORWARD_FILL
    if name == "bfill":
        return TraceRule.BACKWARD_FILL
    return TraceRule.PASSTHROUGH


def causal_history_requirements(specification: LogicalGraphSpec | Mapping[str, Any]) -> dict[str, int | None]:
    """Prove preceding observation counts for finite built-in dependency closures.

    Zero means same-coordinate processing; None requires the full history. A
    count is per asset's admitted coordinates, NOT calendar sessions: dynamic
    Universe gaps require a membership-based halo proof by the caller. This
    helper never proves input-prefix integrity or stateful checkpoint reuse.
    """
    from bagelquant_core.logical import LogicalGraphSpec
    from bagelquant_core.operator import OPERATOR_REGISTRY

    spec = (specification if isinstance(specification, LogicalGraphSpec)
            else LogicalGraphSpec.from_dict(specification))
    result: dict[str, int | None] = {}
    for node in spec.nodes:
        if node.node_type == "input":
            result[node.node_id] = 0
            continue
        operator = OPERATOR_REGISTRY.get(node.operator)
        own = operator.history(node.parameters)
        parents = [result[parent] for parent in (*node.inputs,
            *(parent for values in node.panel_parameters.values() for parent in values))]
        result[node.node_id] = (None if own is None or any(value is None for value in parents)
            else own + max(parents, default=0))
    return result


def _finite_history(operation: Callable[..., Any], parameters: Mapping[str, Any]) -> int | None:
    module, name = operation.__module__, operation.__name__
    if not module.startswith("bagelquant_core."):
        return None
    if name == "smooth" and module == "bagelquant_core.operator.rolling_stats":
        from bagelquant_core.operator.rolling_stats import _SMOOTH_WINDOW
        return _SMOOTH_WINDOW - 1
    if name in {"lag", "diff", "pct_change"}:
        return _history_parameter(parameters.get("periods", 1))
    if name == "remove_repeated":
        return 1
    windows = {"rolling_mean", "rolling_std", "rolling_min", "rolling_max", "rolling_sum",
        "rolling_var", "rolling_median", "rolling_skew", "rolling_kurt", "rolling_percentile",
        "rolling_rank", "rolling_zscore", "rolling_ewm_fw", "date_age_constraint",
        "rolling_corr", "rolling_cov"}
    if name in windows:
        value = _history_parameter(parameters.get("window"))
        return None if value is None else max(0, value - 1)
    if name in {"rolling_ols", "rolling_ridge", "rolling_lasso", "rolling_elastic_net"}:
        # Regressions fit strictly prior windows, then predict the current row.
        return _history_parameter(parameters.get("window"))
    pointwise_modules = {"basic", "boxcox", "logarithmic", "outlier", "normalization",
        "power", "ranking", "replace", "sign", "translation", "trigonometric",
        "variance_stabilization"}
    if module.startswith("bagelquant_core.operator.") and module.rsplit(".", 1)[-1] in pointwise_modules:
        return 0
    if module == "bagelquant_core.operator.temporal" and name in {
        "canonicalize_values", "project_domain", "notnan", "denoise", "posonly", "negonly",
        "constant", "replace_inf"}:
        return 0
    if module == "bagelquant_core.operator.missing" and name in {"fillna", "fillna_zero"}:
        return 0
    if module.startswith("bagelquant_core.operator.") and module.rsplit(".", 1)[-1] in {
        "arithmetic", "aggregation", "comparison", "cross_sectional", "scaling"}:
        return 0
    if module == "bagelquant_core.operator.combination" and name in {"project", "mask", "coalesce"}:
        return 0
    return None


def _history_parameter(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


__all__ = [
    "ExecutionMode",
    "InputDensity",
    "OperationContract",
    "TraceFunction",
    "TraceRule",
    "default_operation_contract",
    "causal_history_requirements",
]
