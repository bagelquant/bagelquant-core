"""Strongly typed AlphaValue-to-prediction composition.

Prediction composers consume AlphaValue panels that have already been aligned
and standardized by an Alpha Policy and produce a :class:`PredictionPanel`.
That typed result may pass through Transformer nodes while remaining a
PredictionPanel; it cannot enter another Composer or serve as an auxiliary
Panel parameter.  Supervised composers may additionally consume an explicit
target/availability context supplied by a higher-level backtesting package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
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


@dataclass(frozen=True, slots=True)
class FamaMacBethOLSResult:
    """Prediction and auditable rolling cross-sectional OLS diagnostics."""

    prediction: pl.DataFrame
    factor_returns: pl.DataFrame
    rolling_premia: pl.DataFrame
    period_diagnostics: pl.DataFrame


PREDICTION_COMPOSER_REGISTRY: Registry[type["PredictionComposer"]] = Registry(
    "prediction composer"
)

_PREDICTION_CONTRACT = OperationContract(
    execution=ExecutionMode.EAGER_BARRIER,
    density=InputDensity.SPARSE_OK,
    trace_rule=TraceRule.PARENT_MAX,
)


class PredictionComposer(ABC):
    """Base class for allowlisted PredictionPanel graph composers."""

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

    @property
    def quantiles(self) -> int | None:
        return None

    @property
    def half_life(self) -> int | None:
        return None

    def compose(
        self,
        *alpha_values: "Panel | Graph[Panel]",
        training: PredictionTrainingContext | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Graph[PredictionPanel]":
        """Build a typed prediction-composer graph."""

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
        if self._composer.quantiles is not None:
            result["quantiles"] = self._composer.quantiles
        if self._composer.half_life is not None:
            result["half_life"] = self._composer.half_life
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
        return _rank_weighted_prediction(
            wide,
            targets,
            availability,
            window=self.window,
            metric="spearman",
        )


class ICWeightedDecayPredictionComposer(_WindowPredictionComposer):
    """Combine AlphaValues with positive exponentially weighted Spearman IC."""

    kind = "ic_weighted_decay"

    def __init__(self, window: int, half_life: int) -> None:
        super().__init__(window)
        if (
            not isinstance(half_life, int)
            or isinstance(half_life, bool)
            or half_life <= 0
        ):
            raise ValueError("prediction composer half_life must be a positive integer")
        self._half_life = half_life

    @property
    def half_life(self) -> int:
        return self._half_life

    def _compose_supervised(
        self,
        wide: pl.DataFrame,
        targets: pl.DataFrame,
        availability: pl.DataFrame,
    ) -> pl.DataFrame:
        return _rank_weighted_prediction(
            wide,
            targets,
            availability,
            window=self.window,
            metric="spearman",
            half_life=self.half_life,
        )


class QuantileICWeightedPredictionComposer(_WindowPredictionComposer):
    """Combine AlphaValues with positive rolling mean quantile-rank IC."""

    kind = "quantile_ic_weighted"

    def __init__(self, window: int, quantiles: int) -> None:
        super().__init__(window)
        if (
            not isinstance(quantiles, int)
            or isinstance(quantiles, bool)
            or quantiles < 2
        ):
            raise ValueError("prediction composer quantiles must be at least 2")
        self._quantiles = quantiles

    @property
    def quantiles(self) -> int:
        return self._quantiles

    def _compose_supervised(
        self,
        wide: pl.DataFrame,
        targets: pl.DataFrame,
        availability: pl.DataFrame,
    ) -> pl.DataFrame:
        return _rank_weighted_prediction(
            wide,
            targets,
            availability,
            window=self.window,
            metric="quantile_rank",
            quantiles=self.quantiles,
        )


class OLSPredictionComposer(_WindowPredictionComposer):
    """Average rolling per-period cross-sectional OLS coefficients."""

    kind = "ols"

    def _compose_supervised(
        self,
        wide: pl.DataFrame,
        targets: pl.DataFrame,
        availability: pl.DataFrame,
    ) -> pl.DataFrame:
        return _fama_macbeth_ols_from_wide(
            wide,
            targets,
            availability,
            window=self.window,
            factor_names=tuple(_alpha_names(wide)),
        ).prediction


class GLSPredictionComposer(_WindowPredictionComposer):
    """Combine per-period OLS estimates by inverse coefficient covariance."""

    kind = "gls"

    def _compose_supervised(
        self,
        wide: pl.DataFrame,
        targets: pl.DataFrame,
        availability: pl.DataFrame,
    ) -> pl.DataFrame:
        return _gls_prediction(
            wide,
            targets,
            availability,
            window=self.window,
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


def quantile_rank_information_coefficient(
    values: Iterable[object],
    targets: Iterable[object],
    *,
    quantiles: int,
) -> float | None:
    """Return the monotonic IC of equal-count quantile mean target returns.

    Quantile membership is determined from every finite value before target
    availability is considered.  The first quantile contains the highest
    values, so monotonically descending q1-to-qN target means produce ``+1``.
    """

    if (
        not isinstance(quantiles, int)
        or isinstance(quantiles, bool)
        or quantiles < 2
    ):
        raise ValueError("quantiles must be an integer of at least 2")
    value_array = np.asarray(tuple(values), dtype=float)
    target_array = np.asarray(tuple(targets), dtype=float)
    if value_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("values and targets must be one-dimensional")
    if value_array.size != target_array.size:
        raise ValueError("values and targets must have equal lengths")
    finite_value_positions = np.flatnonzero(np.isfinite(value_array))
    count = int(finite_value_positions.size)
    if count < quantiles:
        return None
    ordered_positions = finite_value_positions[
        np.argsort(-value_array[finite_value_positions], kind="stable")
    ]
    buckets = (
        np.floor(np.arange(count, dtype=float) * quantiles / count).astype(int) + 1
    )
    group_returns = np.full(quantiles, np.nan, dtype=float)
    for bucket in range(1, quantiles + 1):
        members = ordered_positions[buckets == bucket]
        finite_targets = target_array[members]
        finite_targets = finite_targets[np.isfinite(finite_targets)]
        if finite_targets.size == 0:
            return None
        group_returns[bucket - 1] = float(finite_targets.mean())
    if np.unique(group_returns).size < 2:
        return None
    high_factor_scores = np.arange(quantiles, 0, -1, dtype=float)
    result = _spearman(high_factor_scores, group_returns)
    return result if np.isfinite(result) else None


def _rank_weighted_prediction(
    wide: pl.DataFrame,
    targets: pl.DataFrame,
    availability: pl.DataFrame,
    *,
    window: int,
    metric: str,
    quantiles: int | None = None,
    half_life: int | None = None,
) -> pl.DataFrame:
    data = _join_training(wide, targets, availability)
    alpha_names = _alpha_names(wide)
    records: list[tuple[date, int, np.ndarray]] = []
    for period in _training_periods(data):
        sample = data.filter(pl.col(TIME) == period)
        metric_values = np.full(len(alpha_names), np.nan)
        target = _float_array(sample, "target")
        for index, column in enumerate(alpha_names):
            values = _float_array(sample, column)
            if metric == "quantile_rank":
                if quantiles is None:
                    raise AssertionError("quantile-rank metric requires quantiles")
                result = quantile_rank_information_coefficient(
                    values,
                    target,
                    quantiles=quantiles,
                )
                if result is not None:
                    metric_values[index] = result
                continue
            valid = np.isfinite(values) & np.isfinite(target)
            if (
                valid.sum() > 1
                and np.unique(values[valid]).size > 1
                and np.unique(target[valid]).size > 1
            ):
                metric_values[index] = _spearman(values[valid], target[valid])
        records.append((period, _period_availability(sample), metric_values))

    rows: list[dict[str, object]] = []
    all_periods = _prediction_periods(wide)
    for period in all_periods:
        history = _available_history(records, period, window, all_periods)
        if history is None:
            continue
        matrix = np.stack([item[2] for item in history])
        valid_alpha = np.isfinite(matrix).all(axis=0)
        weights = np.zeros(len(alpha_names), dtype=float)
        if half_life is None:
            metric_means = matrix[:, valid_alpha].mean(axis=0)
        else:
            ages = np.arange(window - 1, -1, -1, dtype=float)
            period_weights = np.power(2.0, -ages / float(half_life))
            metric_means = np.average(
                matrix[:, valid_alpha],
                axis=0,
                weights=period_weights,
            )
        weights[valid_alpha] = np.maximum(metric_means, 0.0)
        if weights.sum() <= 0:
            continue
        current = wide.filter(pl.col(TIME) == period)
        for item in current.iter_rows(named=True):
            values = np.array(
                [_finite_or_nan(item[name]) for name in alpha_names]
            )
            usable = np.isfinite(values) & (weights > 0)
            denominator = weights[usable].sum()
            if denominator > 0:
                rows.append(
                    {
                        TIME: period,
                        ASSET_ID: item[ASSET_ID],
                        VALUE: float(
                            np.dot(values[usable], weights[usable]) / denominator
                        ),
                    }
                )
    return _rows_frame(rows)


def _gls_prediction(
    wide: pl.DataFrame,
    targets: pl.DataFrame,
    availability: pl.DataFrame,
    *,
    window: int,
) -> pl.DataFrame:
    data = _join_training(wide, targets, availability)
    alpha_names = _alpha_names(wide)
    records, _ = _cross_sectional_regression_records(data, alpha_names)

    rows: list[dict[str, object]] = []
    all_periods = _prediction_periods(wide)
    for period in _prediction_periods(wide):
        history = _available_history(records, period, window, all_periods)
        if history is None or any(item[2] is None for item in history):
            continue
        estimates = [item[2] for item in history]
        assert all(estimate is not None for estimate in estimates)
        try:
            precisions = [
                np.linalg.inv(estimate[1])
                for estimate in estimates
                if estimate is not None
            ]
            precision = np.sum(precisions, axis=0)
            beta = np.linalg.inv(precision) @ np.sum(
                [
                    weight @ estimate[0]
                    for weight, estimate in zip(
                        precisions, estimates, strict=True
                    )
                    if estimate is not None
                ],
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


def fama_macbeth_ols_prediction(
    alpha_values: Mapping[str, pl.DataFrame],
    targets: pl.DataFrame,
    availability: pl.DataFrame,
    *,
    window: int,
) -> FamaMacBethOLSResult:
    """Return rolling Fama--MacBeth OLS predictions and diagnostics.

    Each completed source period is fit independently with an intercept. The
    prediction at a later period uses the arithmetic mean coefficient vector
    from the latest complete, contiguous, and causally available ``window``.
    """

    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("Fama-MacBeth OLS window must be a positive integer")
    if len(alpha_values) < 2:
        raise ValueError("Fama-MacBeth OLS requires at least two AlphaValues")
    factor_names = tuple(alpha_values)
    if any(not isinstance(name, str) or not name.strip() for name in factor_names):
        raise ValueError("Fama-MacBeth factor names must be non-blank strings")
    if "intercept" in factor_names:
        raise ValueError("Fama-MacBeth factor name 'intercept' is reserved")
    wide = _prediction_wide(tuple(alpha_values.values()))
    return _fama_macbeth_ols_from_wide(
        wide,
        targets,
        availability,
        window=window,
        factor_names=factor_names,
    )


def _fama_macbeth_ols_from_wide(
    wide: pl.DataFrame,
    targets: pl.DataFrame,
    availability: pl.DataFrame,
    *,
    window: int,
    factor_names: tuple[str, ...],
) -> FamaMacBethOLSResult:
    data = _join_training(wide, targets, availability)
    alpha_names = _alpha_names(wide)
    if len(alpha_names) != len(factor_names):
        raise ValueError("Fama-MacBeth factor names do not match AlphaValues")
    records, period_diagnostics = _cross_sectional_regression_records(
        data,
        alpha_names,
    )
    terms = ("intercept", *factor_names)
    factor_rows: list[dict[str, object]] = []
    for period, available_at, estimate in records:
        if estimate is None:
            continue
        beta, covariance = estimate
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        for term, coefficient, standard_error in zip(
            terms,
            beta,
            standard_errors,
            strict=True,
        ):
            factor_rows.append(
                {
                    TIME: period,
                    "available_at": date.fromordinal(available_at),
                    "factor": term,
                    "coefficient": float(coefficient),
                    "cross_sectional_standard_error": float(standard_error),
                }
            )

    prediction_rows: list[dict[str, object]] = []
    rolling_rows: list[dict[str, object]] = []
    all_periods = _prediction_periods(wide)
    for period in all_periods:
        history = _available_history(records, period, window, all_periods)
        if history is None or any(item[2] is None for item in history):
            continue
        estimates = [item[2] for item in history]
        assert all(estimate is not None for estimate in estimates)
        beta = np.stack(
            [estimate[0] for estimate in estimates if estimate is not None]
        ).mean(axis=0)
        for term, coefficient in zip(terms, beta, strict=True):
            rolling_rows.append(
                {
                    TIME: period,
                    "window_start": history[0][0],
                    "window_end": history[-1][0],
                    "factor": term,
                    "mean_coefficient": float(coefficient),
                }
            )
        current = wide.filter(pl.col(TIME) == period)
        for item in current.iter_rows(named=True):
            values = np.array([_finite_or_nan(item[name]) for name in alpha_names])
            if np.isfinite(values).all():
                prediction_rows.append(
                    {
                        TIME: period,
                        ASSET_ID: item[ASSET_ID],
                        VALUE: float(beta[0] + np.dot(values, beta[1:])),
                    }
                )
    return FamaMacBethOLSResult(
        prediction=_rows_frame(prediction_rows),
        factor_returns=_fama_macbeth_factor_frame(factor_rows),
        rolling_premia=_fama_macbeth_rolling_frame(rolling_rows),
        period_diagnostics=period_diagnostics,
    )


def _cross_sectional_regression_records(
    data: pl.DataFrame,
    alpha_names: list[str],
) -> tuple[
    list[tuple[date, int, tuple[np.ndarray, np.ndarray] | None]],
    pl.DataFrame,
]:
    coefficient_count = len(alpha_names) + 1
    records: list[tuple[date, int, tuple[np.ndarray, np.ndarray] | None]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for period in _training_periods(data):
        sample = data.filter(pl.col(TIME) == period)
        features = np.column_stack(
            [_float_array(sample, name) for name in alpha_names]
        )
        target = _float_array(sample, "target")
        valid = np.isfinite(target) & np.isfinite(features).all(axis=1)
        observation_count = int(valid.sum())
        available_at = _period_availability(sample)
        estimate: tuple[np.ndarray, np.ndarray] | None = None
        rank: int | None = None
        condition_number: float | None = None
        r_squared: float | None = None
        residual_standard_error: float | None = None
        status = "insufficient_observations"
        if observation_count:
            design = np.column_stack([np.ones(observation_count), features[valid]])
            rank = int(np.linalg.matrix_rank(design))
            condition = float(np.linalg.cond(design))
            condition_number = condition if np.isfinite(condition) else None
        if observation_count > coefficient_count:
            if rank != coefficient_count:
                status = "rank_deficient"
            else:
                gram = design.T @ design
                try:
                    inverse = np.linalg.inv(gram)
                except np.linalg.LinAlgError:
                    inverse = None
                    status = "singular"
                if inverse is not None:
                    beta = inverse @ design.T @ target[valid]
                    residual = target[valid] - design @ beta
                    sigma2 = float(
                        np.dot(residual, residual)
                        / (observation_count - coefficient_count)
                    )
                    covariance = sigma2 * inverse
                    target_deviation = target[valid] - target[valid].mean()
                    total_sum_squares = float(
                        np.dot(target_deviation, target_deviation)
                    )
                    if total_sum_squares > 0:
                        r_squared = float(
                            1.0
                            - np.dot(residual, residual) / total_sum_squares
                        )
                    residual_standard_error = float(np.sqrt(max(sigma2, 0.0)))
                    if np.isfinite(beta).all() and np.isfinite(covariance).all():
                        estimate = (beta, covariance)
                        status = "complete"
                    else:
                        status = "non_finite"
        records.append((period, available_at, estimate))
        diagnostic_rows.append(
            {
                TIME: period,
                "available_at": date.fromordinal(available_at),
                "status": status,
                "observation_count": observation_count,
                "feature_count": len(alpha_names),
                "rank": rank,
                "condition_number": condition_number,
                "r_squared": r_squared,
                "residual_standard_error": residual_standard_error,
            }
        )
    return records, _fama_macbeth_period_frame(diagnostic_rows)


def _fama_macbeth_factor_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        TIME: pl.Date,
        "available_at": pl.Date,
        "factor": pl.String,
        "coefficient": pl.Float64,
        "cross_sectional_standard_error": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(TIME, "factor")


def _fama_macbeth_rolling_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        TIME: pl.Date,
        "window_start": pl.Date,
        "window_end": pl.Date,
        "factor": pl.String,
        "mean_coefficient": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(TIME, "factor")


def _fama_macbeth_period_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        TIME: pl.Date,
        "available_at": pl.Date,
        "status": pl.String,
        "observation_count": pl.Int64,
        "feature_count": pl.Int64,
        "rank": pl.Int64,
        "condition_number": pl.Float64,
        "r_squared": pl.Float64,
        "residual_standard_error": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(TIME)


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
    ICWeightedDecayPredictionComposer,
    QuantileICWeightedPredictionComposer,
    OLSPredictionComposer,
    GLSPredictionComposer,
):
    PREDICTION_COMPOSER_REGISTRY.add(_composer_type.kind, _composer_type)


__all__ = [
    "PREDICTION_COMPOSER_REGISTRY",
    "EqualWeightPredictionComposer",
    "FamaMacBethOLSResult",
    "GLSPredictionComposer",
    "ICWeightedDecayPredictionComposer",
    "ICWeightedPredictionComposer",
    "IdentityPredictionComposer",
    "OLSPredictionComposer",
    "PredictionComposer",
    "PredictionTrainingContext",
    "QuantileICWeightedPredictionComposer",
    "fama_macbeth_ols_prediction",
    "quantile_rank_information_coefficient",
]
