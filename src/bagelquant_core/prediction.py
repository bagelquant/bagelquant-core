"""Strongly typed AlphaValue-to-prediction composition.

Prediction composers are terminal graph operations.  They consume AlphaValue
panels that have already been aligned and standardized by an Alpha Policy and
produce a :class:`PredictionPanel`.  Supervised composers may additionally
consume an explicit target/availability context supplied by a higher-level
backtesting package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import numpy as np
import polars as pl

from ._operation import as_node
from .frame import ASSET_ID, TIME, VALUE
from .node import Node
from .operation_contract import (
    ExecutionMode,
    InputDensity,
    OperationContract,
    TraceRule,
)
from .registry import Registry

if TYPE_CHECKING:
    from .graph import Graph
    from .panel import Panel, PredictionPanel


@dataclass(frozen=True, slots=True)
class PredictionTrainingContext:
    """Supervised targets keyed by source period and their availability dates.

    ``targets`` contains the realized next-period return at the AlphaValue
    observation key. ``availability`` contains the corresponding availability
    date encoded as ``date.toordinal()`` in its value column.  Keeping both as
    Panels makes the training dependency explicit and graph-serializable.
    """

    targets: "Panel | Graph[Panel]"
    availability: "Panel | Graph[Panel]"


PREDICTION_COMPOSER_REGISTRY: Registry[type["PredictionComposer"]] = Registry(
    "prediction composer"
)

_PREDICTION_CONTRACT = OperationContract(
    execution=ExecutionMode.EAGER_BARRIER,
    density=InputDensity.SPARSE_OK,
    trace_rule=TraceRule.PARENT_MAX,
)


class PredictionComposer(ABC):
    """Base class for terminal, allowlisted PredictionPanel graph composers."""

    kind: ClassVar[str]
    supervised: ClassVar[bool] = False

    @property
    def display_name(self) -> str:
        return self.kind

    @property
    def operation(self):
        """Expose the frame callable used by execution identity inspection."""

        return self._compute_frames

    @property
    def contract(self) -> OperationContract:
        return _PREDICTION_CONTRACT

    @property
    def window(self) -> int | None:
        return None

    def compose(
        self,
        *alpha_values: "Panel | Graph[Panel]",
        training: PredictionTrainingContext | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Graph[PredictionPanel]":
        """Build a terminal prediction-composer graph."""

        from .graph import Graph

        if not alpha_values:
            raise ValueError("prediction composer requires at least one AlphaValue")
        self._validate_alpha_count(len(alpha_values))
        if self.supervised and training is None:
            raise ValueError(f"{self.kind} requires a PredictionTrainingContext")
        if not self.supervised and training is not None:
            raise ValueError(f"{self.kind} does not accept a training context")
        sources = list(alpha_values)
        if training is not None:
            sources.extend((training.targets, training.availability))
        return Graph._from_nodes(
            (
                _PredictionComposerNode(
                    parents=tuple(
                        as_node(source, kind="PredictionComposer") for source in sources
                    ),
                    composer=self,
                    alpha_count=len(alpha_values),
                    name=name or self.kind,
                    metadata=metadata,
                ),
            )
        )

    __call__ = compose

    @abstractmethod
    def _validate_alpha_count(self, count: int) -> None: ...

    def _compute_frames(
        self,
        *frames: pl.DataFrame,
        alpha_count: int,
    ) -> pl.DataFrame:
        alphas = tuple(frames[:alpha_count])
        wide = _prediction_wide(alphas)
        if self.supervised:
            if len(frames) != alpha_count + 2:
                raise ValueError("supervised prediction composer inputs are incomplete")
            return self._compose_supervised(wide, frames[-2], frames[-1])
        if len(frames) != alpha_count:
            raise ValueError("unsupervised prediction composer received training inputs")
        return self._compose_unsupervised(wide)

    def _compose_unsupervised(self, wide: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def _compose_supervised(
        self,
        wide: pl.DataFrame,
        targets: pl.DataFrame,
        availability: pl.DataFrame,
    ) -> pl.DataFrame:
        raise NotImplementedError


class _PredictionComposerNode(Node):
    node_type = "prediction_composer"

    def __init__(
        self,
        *,
        parents: tuple[Node, ...],
        composer: PredictionComposer,
        alpha_count: int,
        name: str,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        super().__init__(name=name, metadata=metadata)
        self._parents = parents
        self._composer = composer
        self._alpha_count = alpha_count

    @property
    def parents(self) -> tuple[Node, ...]:
        return self._parents

    @property
    def operation(self) -> PredictionComposer:
        return self._composer

    @property
    def contract(self) -> OperationContract:
        return self._composer.contract

    def compute(self, *frames: pl.DataFrame) -> pl.DataFrame:
        return self._composer._compute_frames(
            *frames,
            alpha_count=self._alpha_count,
        )

    def config(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "prediction_composer": self._composer.kind,
            "alpha_count": self._alpha_count,
        }
        if self._composer.window is not None:
            result["window"] = self._composer.window
        return result


class IdentityPredictionComposer(PredictionComposer):
    """Pass one policy-processed AlphaValue through unchanged."""

    kind = "identity"

    def _validate_alpha_count(self, count: int) -> None:
        if count != 1:
            raise ValueError("identity prediction composer requires exactly one AlphaValue")

    def _compose_unsupervised(self, wide: pl.DataFrame) -> pl.DataFrame:
        return _prediction_output(wide, pl.col("alpha_0"))


class EqualWeightPredictionComposer(PredictionComposer):
    """Average every available policy-processed AlphaValue per asset."""

    kind = "equal_weight"

    def _validate_alpha_count(self, count: int) -> None:
        if count < 2:
            raise ValueError("equal-weight prediction composer requires at least two AlphaValues")

    def _compose_unsupervised(self, wide: pl.DataFrame) -> pl.DataFrame:
        columns = _alpha_columns(wide)
        return _prediction_output(wide, pl.mean_horizontal(*columns))


class _WindowPredictionComposer(PredictionComposer):
    supervised = True

    def __init__(self, window: int) -> None:
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError("prediction composer window must be a positive integer")
        self._window = window

    @property
    def window(self) -> int:
        return self._window

    def _validate_alpha_count(self, count: int) -> None:
        if count < 2:
            raise ValueError(f"{self.kind} requires at least two AlphaValues")


class ICWeightedPredictionComposer(_WindowPredictionComposer):
    """Combine AlphaValues with positive rolling mean Spearman IC."""

    kind = "ic_weighted"

    def _compose_supervised(
        self,
        wide: pl.DataFrame,
        targets: pl.DataFrame,
        availability: pl.DataFrame,
    ) -> pl.DataFrame:
        data = _join_training(wide, targets, availability)
        alpha_names = _alpha_names(wide)
        records: list[tuple[date, int, np.ndarray]] = []
        for period in _training_periods(data):
            sample = data.filter(pl.col(TIME) == period)
            ic_values = np.full(len(alpha_names), np.nan)
            target = _float_array(sample, "target")
            for index, column in enumerate(alpha_names):
                values = _float_array(sample, column)
                valid = np.isfinite(values) & np.isfinite(target)
                if valid.sum() > 1 and np.unique(values[valid]).size > 1 and np.unique(target[valid]).size > 1:
                    ic_values[index] = _spearman(values[valid], target[valid])
            records.append((period, _period_availability(sample), ic_values))

        rows: list[dict[str, object]] = []
        all_periods = _prediction_periods(wide)
        for period in _prediction_periods(wide):
            history = _available_history(records, period, self.window, all_periods)
            if history is None:
                continue
            matrix = np.stack([item[2] for item in history])
            valid_alpha = np.isfinite(matrix).all(axis=0)
            weights = np.zeros(len(alpha_names), dtype=float)
            weights[valid_alpha] = np.maximum(matrix[:, valid_alpha].mean(axis=0), 0.0)
            if weights.sum() <= 0:
                continue
            current = wide.filter(pl.col(TIME) == period)
            for item in current.iter_rows(named=True):
                values = np.array([_finite_or_nan(item[name]) for name in alpha_names])
                usable = np.isfinite(values) & (weights > 0)
                denominator = weights[usable].sum()
                if denominator > 0:
                    rows.append({TIME: period, ASSET_ID: item[ASSET_ID], VALUE: float(np.dot(values[usable], weights[usable]) / denominator)})
        return _rows_frame(rows)


class OLSPredictionComposer(_WindowPredictionComposer):
    """Average rolling per-period cross-sectional OLS coefficients."""

    kind = "ols"

    def _compose_supervised(
        self,
        wide: pl.DataFrame,
        targets: pl.DataFrame,
        availability: pl.DataFrame,
    ) -> pl.DataFrame:
        return _regression_prediction(
            wide,
            targets,
            availability,
            window=self.window,
            method="ols",
        )


class GLSPredictionComposer(_WindowPredictionComposer):
    """Combine per-period OLS estimates by inverse coefficient covariance."""

    kind = "gls"

    def _compose_supervised(
        self,
        wide: pl.DataFrame,
        targets: pl.DataFrame,
        availability: pl.DataFrame,
    ) -> pl.DataFrame:
        return _regression_prediction(
            wide,
            targets,
            availability,
            window=self.window,
            method="gls",
        )


def _prediction_wide(frames: tuple[pl.DataFrame, ...]) -> pl.DataFrame:
    if not frames:
        raise ValueError("prediction composer requires AlphaValue frames")
    keys = pl.concat([frame.select(TIME, ASSET_ID) for frame in frames]).unique().sort([TIME, ASSET_ID])
    wide = keys
    for index, frame in enumerate(frames):
        wide = wide.join(
            frame.select(TIME, ASSET_ID, pl.col(VALUE).alias(f"alpha_{index}")),
            on=[TIME, ASSET_ID],
            how="left",
        )
    expressions = [
        pl.when(pl.col(name).fill_nan(None).is_finite())
        .then(pl.col(name).fill_nan(None))
        .otherwise(None)
        .alias(name)
        for name in _alpha_names(wide)
    ]
    return wide.with_columns(expressions).sort([TIME, ASSET_ID])


def _prediction_output(frame: pl.DataFrame, expression: pl.Expr) -> pl.DataFrame:
    return (
        frame.select(TIME, ASSET_ID, expression.alias(VALUE))
        .drop_nulls(VALUE)
        .sort([TIME, ASSET_ID])
    )


def _join_training(
    wide: pl.DataFrame,
    targets: pl.DataFrame,
    availability: pl.DataFrame,
) -> pl.DataFrame:
    return (
        wide.join(
            targets.select(TIME, ASSET_ID, pl.col(VALUE).alias("target")),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .join(
            availability.select(TIME, ASSET_ID, pl.col(VALUE).alias("available_at")),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .sort([TIME, ASSET_ID])
    )


def _regression_prediction(
    wide: pl.DataFrame,
    targets: pl.DataFrame,
    availability: pl.DataFrame,
    *,
    window: int,
    method: str,
) -> pl.DataFrame:
    data = _join_training(wide, targets, availability)
    alpha_names = _alpha_names(wide)
    coefficient_count = len(alpha_names) + 1
    records: list[tuple[date, int, tuple[np.ndarray, np.ndarray] | None]] = []
    for period in _training_periods(data):
        sample = data.filter(pl.col(TIME) == period)
        features = np.column_stack([_float_array(sample, name) for name in alpha_names])
        target = _float_array(sample, "target")
        valid = np.isfinite(target) & np.isfinite(features).all(axis=1)
        estimate: tuple[np.ndarray, np.ndarray] | None = None
        if valid.sum() > coefficient_count:
            design = np.column_stack([np.ones(valid.sum()), features[valid]])
            if np.linalg.matrix_rank(design) == coefficient_count:
                gram = design.T @ design
                try:
                    inverse = np.linalg.inv(gram)
                except np.linalg.LinAlgError:
                    inverse = None
                if inverse is not None:
                    beta = inverse @ design.T @ target[valid]
                    residual = target[valid] - design @ beta
                    sigma2 = float(np.dot(residual, residual) / (valid.sum() - coefficient_count))
                    covariance = sigma2 * inverse
                    if np.isfinite(beta).all() and np.isfinite(covariance).all():
                        estimate = (beta, covariance)
        records.append((period, _period_availability(sample), estimate))

    rows: list[dict[str, object]] = []
    all_periods = _prediction_periods(wide)
    for period in _prediction_periods(wide):
        history = _available_history(records, period, window, all_periods)
        if history is None or any(item[2] is None for item in history):
            continue
        estimates = [item[2] for item in history]
        assert all(estimate is not None for estimate in estimates)
        if method == "ols":
            beta = np.stack([estimate[0] for estimate in estimates if estimate is not None]).mean(axis=0)
        else:
            try:
                precisions = [np.linalg.inv(estimate[1]) for estimate in estimates if estimate is not None]
                precision = np.sum(precisions, axis=0)
                beta = np.linalg.inv(precision) @ np.sum(
                    [weight @ estimate[0] for weight, estimate in zip(precisions, estimates, strict=True) if estimate is not None],
                    axis=0,
                )
            except np.linalg.LinAlgError:
                continue
        current = wide.filter(pl.col(TIME) == period)
        for item in current.iter_rows(named=True):
            values = np.array([_finite_or_nan(item[name]) for name in alpha_names])
            if np.isfinite(values).all():
                rows.append({TIME: period, ASSET_ID: item[ASSET_ID], VALUE: float(beta[0] + np.dot(values, beta[1:]))})
    return _rows_frame(rows)


def _available_history(records, period: date, window: int, all_periods: list[date]):
    eligible = [
        record
        for record in records
        if record[0] < period and record[1] <= period.toordinal()
    ]
    if len(eligible) < window:
        return None
    selected = eligible[-window:]
    positions = {value: index for index, value in enumerate(all_periods)}
    selected_positions = [positions[record[0]] for record in selected]
    if selected_positions != list(
        range(selected_positions[0], selected_positions[0] + window)
    ):
        return None
    return selected


def _period_availability(sample: pl.DataFrame) -> int:
    values = sample.get_column("available_at").drop_nulls().unique()
    if len(values) != 1:
        raise ValueError("training period must have one availability date")
    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError("training availability must be finite")
    return int(value)


def _training_periods(data: pl.DataFrame) -> list[date]:
    return data.filter(pl.col("target").is_not_null() & pl.col("available_at").is_not_null()).get_column(TIME).unique().sort().to_list()


def _prediction_periods(wide: pl.DataFrame) -> list[date]:
    return wide.select(TIME).unique().sort(TIME).get_column(TIME).to_list()


def _alpha_names(frame: pl.DataFrame) -> list[str]:
    return [name for name in frame.columns if name.startswith("alpha_")]


def _alpha_columns(frame: pl.DataFrame) -> list[pl.Expr]:
    return [pl.col(name) for name in _alpha_names(frame)]


def _float_array(frame: pl.DataFrame, column: str) -> np.ndarray:
    return np.array([_finite_or_nan(value) for value in frame.get_column(column)], dtype=float)


def _finite_or_nan(value: object) -> float:
    if value is None:
        return np.nan
    result = float(value)
    return result if np.isfinite(result) else np.nan


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pl.Series(left).rank("average").to_numpy()
    right_rank = pl.Series(right).rank("average").to_numpy()
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _rows_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64})
    return pl.DataFrame(rows, schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64}).sort([TIME, ASSET_ID])


for _composer_type in (
    IdentityPredictionComposer,
    EqualWeightPredictionComposer,
    ICWeightedPredictionComposer,
    OLSPredictionComposer,
    GLSPredictionComposer,
):
    PREDICTION_COMPOSER_REGISTRY.add(_composer_type.kind, _composer_type)


__all__ = [
    "PREDICTION_COMPOSER_REGISTRY",
    "EqualWeightPredictionComposer",
    "GLSPredictionComposer",
    "ICWeightedPredictionComposer",
    "IdentityPredictionComposer",
    "OLSPredictionComposer",
    "PredictionComposer",
    "PredictionTrainingContext",
]
