"""Causal, resumable processing of typed predictions on an explicit calendar.

This is an eager numerical boundary, not a graph operation or storage service.
Callers supply the evaluation calendar and persist checkpoints explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import polars as pl

from bagelquant_core.frame import normalize_date_expression
from bagelquant_core.hashing import hash_mapping
from bagelquant_core.node import Node


@dataclass(frozen=True, slots=True)
class PredictionSmoothingConfig:
    method: Literal["none", "sma", "ewma"] = "none"
    window: int | None = None
    half_life: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"none", "sma", "ewma"}:
            raise ValueError("smoothing method must be none, sma or ewma")
        if self.method == "sma":
            if (
                not isinstance(self.window, int)
                or isinstance(self.window, bool)
                or self.window < 1
            ):
                raise ValueError("SMA window must be a positive integer")
        elif self.window is not None:
            raise ValueError("window is only valid for SMA")
        if self.method == "ewma":
            if (
                not isinstance(self.half_life, (int, float))
                or isinstance(self.half_life, bool)
                or not math.isfinite(self.half_life)
                or self.half_life <= 0
            ):
                raise ValueError("EWMA half_life must be finite and positive")
        elif self.half_life is not None:
            raise ValueError("half_life is only valid for EWMA")


@dataclass(frozen=True, slots=True)
class PredictionSmoothingState:
    """Immutable checkpoint; ``values`` contains only last-period active assets."""

    config: PredictionSmoothingConfig
    last_time: date
    calendar_prefix_hash: str
    values: tuple[tuple[str, tuple[float, ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": {
                "method": self.config.method,
                "window": self.config.window,
                "half_life": self.config.half_life,
            },
            "last_time": self.last_time.isoformat(),
            "calendar_prefix_hash": self.calendar_prefix_hash,
            "values": {asset: list(values) for asset, values in self.values},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PredictionSmoothingState:
        if value.get("version") != 1:
            raise ValueError("unsupported prediction smoothing checkpoint version")
        return cls(
            config=PredictionSmoothingConfig(**value["config"]),
            last_time=date.fromisoformat(value["last_time"]),
            calendar_prefix_hash=str(value["calendar_prefix_hash"]),
            values=tuple(
                (str(asset), tuple(float(item) for item in values))
                for asset, values in sorted(value["values"].items())
            ),
        )


@dataclass(frozen=True, slots=True)
class PredictionSmoothingResult:
    prediction: Node
    state: PredictionSmoothingState


def smooth_prediction(
    prediction: Node,
    config: PredictionSmoothingConfig,
    *,
    evaluation_calendar: Sequence[Any] | pl.Series,
    state: PredictionSmoothingState | None = None,
) -> PredictionSmoothingResult:
    """Smooth a contiguous calendar block, resetting on gaps and Universe exits.

    ``prediction.domain.times`` must be a contiguous block of the supplied
    strictly ordered calendar. A fresh call starts at the calendar's first date;
    a resumed call starts immediately after its checkpoint. Thus callers cannot
    accidentally substitute a visible UI window for the required history. The
    calendar unit is an evaluation period, not an elapsed day. Null, NaN, infinity
    and inactive membership all interrupt continuity without emitting a value.
    """

    if not (isinstance(prediction, Node) and prediction.value_type == "prediction"):
        raise TypeError("smoothing requires a Node")
    if not isinstance(config, PredictionSmoothingConfig):
        raise TypeError("config must be a PredictionSmoothingConfig")
    raw = pl.DataFrame({"time": evaluation_calendar})
    days = raw.select(normalize_date_expression("time", raw.schema["time"]))["time"]
    if (
        days.is_empty()
        or days.null_count()
        or days.n_unique() != len(days)
        or not days.is_sorted()
    ):
        raise ValueError("evaluation_calendar must contain ordered unique valid dates")
    calendar = days.to_list()
    periods = prediction.domain.times.to_list()
    positions = {day: index for index, day in enumerate(calendar)}
    start = positions.get(periods[0])
    if start is None or calendar[start : start + len(periods)] != periods:
        raise ValueError(
            "prediction domain must be a contiguous evaluation-calendar block"
        )

    def prefix_hash(end: int) -> str:
        return hash_mapping({"calendar": [day.isoformat() for day in calendar[:end]]})

    previous: dict[str, tuple[float, ...]] = {}
    if state is None:
        if start != 0:
            raise ValueError(
                "smoothing requires complete warm-up history or a checkpoint"
            )
    else:
        if state.config != config:
            raise ValueError("smoothing checkpoint configuration does not match")
        if start == 0 or calendar[start - 1] != state.last_time:
            raise ValueError("smoothing checkpoint must immediately precede this block")
        if state.calendar_prefix_hash != prefix_hash(start):
            raise ValueError("smoothing checkpoint calendar history has changed")
        for asset, values in state.values:
            limit = config.window if config.method == "sma" else 1
            if (
                asset in previous
                or not values
                or len(values) > limit
                or not all(math.isfinite(value) for value in values)
            ):
                raise ValueError("invalid smoothing checkpoint asset state")
            previous[asset] = values

    finite = (
        prediction.collect(dense=False)
        .select("time", "asset_id", "value")
        .filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
    )
    groups = {
        key[0]: frame
        for key, frame in finite.partition_by("time", as_dict=True).items()
    }
    alpha = (
        -math.expm1(-math.log(2.0) / config.half_life)
        if config.method == "ewma"
        else 1.0
    )
    output: list[tuple[date, str, float]] = []
    for day in periods:
        current: dict[str, tuple[float, ...]] = {}
        cross = groups.get(day)
        if cross is not None:
            for asset, value in cross.select("asset_id", "value").iter_rows():
                history = previous.get(asset, ())
                if config.method == "sma":
                    values = (*history, value)[-config.window :]
                    current[asset] = values
                    if len(values) < config.window:
                        continue
                    result = math.fsum(values) / config.window
                elif config.method == "ewma":
                    result = (
                        value
                        if not history
                        else (1.0 - alpha) * history[-1] + alpha * value
                    )
                    current[asset] = (result,)
                else:
                    result = value
                    current[asset] = (value,)
                output.append((day, asset, result))
        previous = current
    frame = pl.DataFrame(
        output,
        schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64},
        orient="row",
    )
    return PredictionSmoothingResult(
        prediction=Node.from_domain(
            frame, prediction.domain, name=prediction.name, metadata=prediction.metadata
        , value_type="prediction"),
        state=PredictionSmoothingState(
            config=config,
            last_time=periods[-1],
            calendar_prefix_hash=prefix_hash(start + len(periods)),
            values=tuple(sorted(previous.items())),
        ),
    )
