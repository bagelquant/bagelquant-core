"""Causal target weights and calendar-anchored decisions, without accounts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from .operator import Operator, OPERATOR_REGISTRY
from .operation_contract import ExecutionMode, InputDensity, OperationContract, TraceRule
from .optimization import project_capped_simplex, solve_regularized_weights, solve_exposure_weights
from .operator_state import restored_operator_state, save_operator_state

_CONTRACT = OperationContract(execution=ExecutionMode.EAGER_BARRIER,
    density=InputDensity.DENSE_REQUIRED, trace_rule=TraceRule.PARENT_MAX)


def _register(function):
    operation = Operator(function, contract=_CONTRACT, output_type="weights")
    OPERATOR_REGISTRY.add(operation.registry_name, operation)
    return operation


@_register
def top_n(frame: pl.DataFrame, *, count: int = 20) -> pl.DataFrame:
    """Select exactly count finite scores per date; ties use ascending asset ID.

    Selected assets receive one and all other assets zero. An insufficient
    cross-section is entirely unavailable, rather than a smaller portfolio.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    parts = []
    for group in frame.partition_by("time", maintain_order=True):
        valid = group.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
        if valid.height < count:
            parts.append(group.with_columns(pl.lit(None, dtype=pl.Float64).alias("value")))
            continue
        selected = valid.sort(["value", "asset_id"], descending=[True, False]).head(count)["asset_id"]
        parts.append(group.with_columns(pl.col("asset_id").is_in(selected.implode()).cast(pl.Float64).alias("value")))
    return pl.concat(parts) if parts else frame


@_register
def equal_weight(frame: pl.DataFrame) -> pl.DataFrame:
    """Assign equal positive weights to a complete binary selection snapshot."""
    invalid = frame.filter(pl.col("value").is_not_null() & ~pl.col("value").is_in([0.0, 1.0]))
    if invalid.height:
        raise ValueError("equal_weight requires a binary selection (use top_n)")
    return frame.with_columns(
        pl.when(pl.col("value").is_not_null().all().over("time") & (pl.col("value").sum().over("time") > 0))
        .then(pl.col("value") / pl.col("value").sum().over("time"))
        .otherwise(None).alias("value"))


@_register
def regularized_weights(frame: pl.DataFrame, *, concentration_penalty: float = 10.0,
        turnover_penalty: float = 0.1, max_weight: float = 0.04,
        tolerance: float = 1e-7, reference: pl.DataFrame | None = None) -> pl.DataFrame:
    """Optimize long-only weights using preceding target weights as state.

    Optional reference contains historical target snapshots. Account holdings
    and simulated fills are never inputs to this numerical operator.
    """
    if not np.isfinite([concentration_penalty, turnover_penalty, max_weight, tolerance]).all():
        raise ValueError("optimizer parameters must be finite")
    if concentration_penalty <= 0 or turnover_penalty < 0 or not 0 < max_weight <= 1 or tolerance <= 0:
        raise ValueError("invalid optimizer parameters")
    checkpoint = restored_operator_state()
    state: dict[str, float] = {} if checkpoint is None else dict(checkpoint["targets"])
    history = [] if reference is None else reference.sort("time").partition_by("time", maintain_order=True)
    cursor = 0
    parts = []
    for group in frame.sort(["time", "asset_id"]).partition_by("time", maintain_order=True):
        day = group["time"][0]
        if checkpoint and day.isoformat() <= checkpoint["through"]:
            parts.append(group.with_columns(pl.lit(None,dtype=pl.Float64).alias("value")))
            continue
        while cursor < len(history) and history[cursor]["time"][0] < day:
            state = dict(history[cursor].select("asset_id", "value").iter_rows())
            cursor += 1
        valid = group.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
        if valid.height * max_weight < 1 - tolerance:
            parts.append(group.with_columns(pl.lit(None, dtype=pl.Float64).alias("value")))
            continue
        assets = valid["asset_id"].to_list()
        anchors = np.array([state.get(asset, 0.0) for asset in assets])
        solution, _ = solve_regularized_weights(valid["value"].to_numpy(), anchors,
            concentration_penalty=concentration_penalty, turnover_penalty=turnover_penalty,
            max_weight=max_weight, tolerance=tolerance)
        solution = project_capped_simplex(solution, max_weight)
        state = dict(zip(assets, solution.tolist(), strict=True))
        parts.append(group.with_columns(pl.col("asset_id").replace_strict(state, default=0.0, return_dtype=pl.Float64).alias("value")))
    if frame.height:
        save_operator_state({"through":str(frame["time"].max()),"targets":state})
    return pl.concat(parts) if parts else frame


@dataclass(frozen=True)
class ExposureBound:
    lower: float | None
    upper: float | None


@_register
def exposure_constrained_weights(frame: pl.DataFrame, *, exposures: tuple[pl.DataFrame, ...],
        lower_bounds: list[float], upper_bounds: list[float],
        concentration_penalty: float = 10.0, turnover_penalty: float = 0.1,
        max_weight: float = 0.04, max_turnover: float | None = None,
        constraint_tolerance: float = 1e-7, reference: pl.DataFrame | None = None) -> pl.DataFrame:
    """Optimize supplied exposures using historical target weights as state.

    Bounds match the ordered auxiliary panels. Missing required exposure
    coordinates and infeasible constraints fail explicitly.
    """
    if not exposures or len(exposures) != len(lower_bounds) or len(exposures) != len(upper_bounds):
        raise ValueError("each exposure requires one lower and upper bound")
    if not np.isfinite([*lower_bounds, *upper_bounds, concentration_penalty, turnover_penalty, max_weight, constraint_tolerance]).all():
        raise ValueError("optimizer parameters and bounds must be finite")
    if any(low > high for low, high in zip(lower_bounds, upper_bounds)) or concentration_penalty <= 0 or turnover_penalty < 0 or not 0 < max_weight <= 1 or constraint_tolerance <= 0:
        raise ValueError("invalid optimizer parameters or bounds")
    if max_turnover is not None and (not np.isfinite(max_turnover) or max_turnover < 0):
        raise ValueError("max_turnover must be finite and nonnegative")
    bounds = {f"exposure_{index}": ExposureBound(low, high) for index, (low, high) in enumerate(zip(lower_bounds, upper_bounds))}
    enriched = frame
    for index, exposure in enumerate(exposures):
        if exposure.unique(["time", "asset_id"]).height != exposure.height:
            raise ValueError("duplicate exposure coordinates")
        enriched = enriched.join(exposure.select("time", "asset_id", pl.col("value").alias(f"exposure_{index}")), on=["time", "asset_id"], how="left")
    checkpoint = restored_operator_state()
    state, parts = ({} if checkpoint is None else dict(checkpoint["targets"])), []
    history = [] if reference is None else reference.sort("time").partition_by("time", maintain_order=True)
    cursor = 0
    for group in enriched.sort(["time", "asset_id"]).partition_by("time", maintain_order=True):
        day = group["time"][0]
        if checkpoint and day.isoformat() <= checkpoint["through"]:
            parts.append(group.select("time","asset_id").with_columns(pl.lit(None,dtype=pl.Float64).alias("value")))
            continue
        while cursor < len(history) and history[cursor]["time"][0] < day:
            state = dict(history[cursor].select("asset_id", "value").iter_rows())
            cursor += 1
        valid = group.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
        if valid.height * max_weight < 1 - constraint_tolerance:
            parts.append(group.select("time", "asset_id").with_columns(pl.lit(None, dtype=pl.Float64).alias("value")))
            continue
        if valid.filter(~pl.all_horizontal([pl.col(name).is_not_null() & pl.col(name).is_finite() for name in bounds])).height:
            raise ValueError(f"required exposure missing at {day}")
        assets = valid["asset_id"].to_list()
        forced_exit = sum(weight for asset, weight in state.items() if asset not in assets)
        solution, _, _ = solve_exposure_weights(valid["value"].to_numpy(), np.array([state.get(asset, 0) for asset in assets]),
            forced_exit, valid, bounds, day, max_weight=max_weight, max_turnover=max_turnover,
            concentration_penalty=concentration_penalty, turnover_penalty=turnover_penalty, constraint_tolerance=constraint_tolerance)
        if abs(solution.sum() - 1) > constraint_tolerance or solution.max() > max_weight + constraint_tolerance:
            raise ValueError(f"optimizer failed target feasibility at {day}")
        for name, bound in bounds.items():
            exposure = float(valid[name].to_numpy() @ solution)
            if exposure < bound.lower - constraint_tolerance or exposure > bound.upper + constraint_tolerance:
                raise ValueError(f"optimizer failed exposure feasibility at {day}")
        state = dict(zip(assets, solution.tolist(), strict=True))
        parts.append(group.select("time", "asset_id").with_columns(pl.col("asset_id").replace_strict(state, default=0.0, return_dtype=pl.Float64).alias("value")))
    if frame.height:
        save_operator_state({"through":str(frame["time"].max()),"targets":state})
    return pl.concat(parts) if parts else frame


@dataclass(frozen=True)
class PortfolioValue:
    """Complete target snapshots plus explicit hold/unavailable instructions."""
    weights: pl.DataFrame
    decisions: pl.DataFrame


def rebalance_value(weights: pl.DataFrame, *, calendar: pl.DataFrame,
        data_start: str | date, every: int = 5, anchor: str = "data_start",
        coverage_calendar: pl.DataFrame | None = None) -> PortfolioValue:
    """Select decisions from the full calendar, independent of input slices."""
    if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
        raise ValueError("every must be a positive integer")
    if anchor != "data_start":
        raise ValueError("anchor must be data_start")
    start = date.fromisoformat(data_start) if isinstance(data_start, str) else data_start
    sessions = calendar.select(pl.col("time").cast(pl.Date)).unique().filter(pl.col("time") >= start).sort("time")["time"].to_list()
    if not sessions:
        raise ValueError("calendar has no session at or after data_start")
    positions = {day: index for index, day in enumerate(sessions)}
    rows, targets = [], []
    groups = {group["time"][0]:group for group in weights.sort(["time", "asset_id"]).partition_by("time", maintain_order=True)}
    coverage = sorted(groups) if coverage_calendar is None else coverage_calendar["time"].unique().sort().to_list()
    for day in coverage:
        group = groups.get(day)
        if day not in positions:
            raise ValueError(f"weight date {day} is absent from the anchored calendar")
        scheduled = positions[day] % every == 0
        available = group is not None and not group.filter(pl.col("value").is_null() | ~pl.col("value").is_finite()).height
        if available and group.filter(pl.col("value") < 0).height:
            raise ValueError("long-only weights cannot be negative")
        status = "hold" if not scheduled else "rebalance" if available else "unavailable"
        rows.append({"time": day, "status": status,
                     "reason": ("empty_universe" if group is None else "insufficient_valid_inputs") if status == "unavailable" else None})
        if status == "rebalance":
            targets.append(group.select("time", "asset_id", "value"))
    return PortfolioValue(pl.concat(targets) if targets else weights.head(0),
        pl.DataFrame(rows, schema={"time": pl.Date, "status": pl.String, "reason": pl.String}))


@_register
def rebalance(frame: pl.DataFrame, *, every: int = 5, anchor: str = "data_start",
        calendar: pl.DataFrame, data_start: str) -> pl.DataFrame:
    """Return complete scheduled target snapshots; store decisions separately."""
    return rebalance_value(frame, calendar=calendar, data_start=data_start, every=every, anchor=anchor).weights
