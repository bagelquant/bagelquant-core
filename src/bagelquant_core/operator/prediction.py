"""Strongly typed AlphaValue-to-prediction composition.

Prediction operators consume numeric feature Nodes and produce prediction Nodes.
Subsequent operators preserve prediction type according to their declared contract.
Supervised operators take separately typed targets and availability dependencies;
all peers and auxiliary roles are recorded in the same graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, replace
from copy import deepcopy
from datetime import date, timedelta
from typing import Any, ClassVar, Mapping

import numpy as np
import polars as pl

from bagelquant_core._operation import as_node
from bagelquant_core.frame import ASSET_ID, TIME, VALUE
from bagelquant_core.node import Node
from bagelquant_core.operation_contract import (
    ExecutionMode,
    InputDensity,
    OperationContract,
    TraceRule,
)
from bagelquant_core.operator import Operator, OPERATOR_REGISTRY
from bagelquant_core.hashing import DATAFRAME_HASH_V1, hash_dataframe
from bagelquant_core.operator_state import (
    checkpoint_capture_enabled, operator_execution_calendar, operator_period_work,
    restored_operator_state, save_operator_state,
)


@dataclass(frozen=True, slots=True)
class PredictionTrainingContext:
    """Supervised targets keyed by source period and their availability dates.

    ``targets`` contains the realized next-period return at the AlphaValue
    observation key. ``availability`` contains the corresponding availability
    date encoded as ``date.toordinal()`` in its value column.  Keeping both as
    Nodes makes the training dependency explicit and graph-serializable.
    """

    targets: Node
    availability: Node


@dataclass(frozen=True, slots=True)
class FamaMacBethOLSResult:
    """Prediction and auditable rolling cross-sectional OLS diagnostics."""

    prediction: pl.DataFrame
    factor_returns: pl.DataFrame
    rolling_premia: pl.DataFrame
    period_diagnostics: pl.DataFrame


_PREDICTION_CONTRACT = OperationContract(
    execution=ExecutionMode.EAGER_BARRIER,
    density=InputDensity.SPARSE_OK,
    trace_rule=TraceRule.PARENT_MAX,
)


class PredictionOperator(ABC):
    """Base class for allowlisted Node graph operators."""

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
        return _prediction_contract(type(self))

    @property
    def window(self) -> int | None:
        return None

    @property
    def quantiles(self) -> int | None:
        return None

    @property
    def half_life(self) -> int | None:
        return None

    def __call__(
        self,
        *alpha_values: Node,
        training: PredictionTrainingContext | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Node:
        """Build a deferred prediction Node."""


        if not alpha_values:
            raise ValueError("prediction operator requires at least one AlphaValue")
        self._validate_alpha_count(len(alpha_values))
        if self.supervised and training is None:
            raise ValueError(f"{self.kind} requires a PredictionTrainingContext")
        if not self.supervised and training is not None:
            raise ValueError(f"{self.kind} does not accept a training context")
        sources = list(alpha_values)
        return Node.from_operation(
                    inputs=tuple(as_node(source, kind="Operator") for source in sources),
                    panel_parameters={} if training is None else {"targets": (as_node(training.targets, kind="Operator"),), "availability": (as_node(training.availability, kind="Operator"),)}, operation=self,
                    config={"alpha_count": len(alpha_values), **{
                        key: value for key, value in (("window", self.window),
                        ("quantiles", self.quantiles), ("half_life", self.half_life))
                        if value is not None}},
                    name=name or self.kind, metadata=metadata,
                )

    @abstractmethod
    def _validate_alpha_count(self, count: int) -> None: ...

    def _compute_frames(self, *alphas: pl.DataFrame, targets=None, availability=None) -> pl.DataFrame:
        wide = _prediction_wide(tuple(alphas))
        if self.supervised:
            if targets is None or availability is None:
                raise ValueError("supervised model requires targets and availability")
            if checkpoint_capture_enabled():
                return _checkpoint_prediction(self, wide, targets, availability)
            return self._compose_supervised(wide, targets, availability)
        if targets is not None or availability is not None:
            raise ValueError("unsupervised model does not accept training inputs")
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


class IdentityPredictionOperator(PredictionOperator):
    """Pass one policy-processed AlphaValue through unchanged."""

    kind = "identity"

    def _validate_alpha_count(self, count: int) -> None:
        if count != 1:
            raise ValueError("identity prediction operator requires exactly one AlphaValue")

    def _compose_unsupervised(self, wide: pl.DataFrame) -> pl.DataFrame:
        return _prediction_output(wide, pl.col("alpha_0"))


class EqualWeightPredictionOperator(PredictionOperator):
    """Average every available policy-processed AlphaValue per asset."""

    kind = "equal_weight"

    def _validate_alpha_count(self, count: int) -> None:
        if count < 2:
            raise ValueError("equal-weight prediction operator requires at least two AlphaValues")

    def _compose_unsupervised(self, wide: pl.DataFrame) -> pl.DataFrame:
        columns = _alpha_columns(wide)
        return _prediction_output(wide, pl.mean_horizontal(*columns))


class _WindowPredictionOperator(PredictionOperator):
    supervised = True

    def __init__(self, window: int) -> None:
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError("prediction operator window must be a positive integer")
        self._window = window

    @property
    def window(self) -> int:
        return self._window

    def _validate_alpha_count(self, count: int) -> None:
        if count < 2:
            raise ValueError(f"{self.kind} requires at least two AlphaValues")


class ICWeightedPredictionOperator(_WindowPredictionOperator):
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


class ICWeightedDecayPredictionOperator(_WindowPredictionOperator):
    """Combine AlphaValues with positive exponentially weighted Spearman IC."""

    kind = "ic_weighted_decay"

    def __init__(self, window: int, half_life: int) -> None:
        super().__init__(window)
        if (
            not isinstance(half_life, int)
            or isinstance(half_life, bool)
            or half_life <= 0
        ):
            raise ValueError("prediction operator half_life must be a positive integer")
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


class QuantileICWeightedPredictionOperator(_WindowPredictionOperator):
    """Combine AlphaValues with positive rolling mean quantile-rank IC."""

    kind = "quantile_ic_weighted"

    def __init__(self, window: int, quantiles: int) -> None:
        super().__init__(window)
        if (
            not isinstance(quantiles, int)
            or isinstance(quantiles, bool)
            or quantiles < 2
        ):
            raise ValueError("prediction operator quantiles must be at least 2")
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


class OLSPredictionOperator(_WindowPredictionOperator):
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


class GLSPredictionOperator(_WindowPredictionOperator):
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
        raise ValueError("prediction operator requires AlphaValue frames")
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
    records = _rank_records(data, alpha_names, metric=metric, quantiles=quantiles)
    return _rank_prediction_from_records(wide, records, window=window,
        all_periods=_prediction_periods(wide), half_life=half_life)


def _rank_records(data, alpha_names, *, metric, quantiles=None):
    """Single per-period IC primitive for initialization and continuation."""
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
    return records


def _rank_prediction_from_records(wide, records, *, window, all_periods, half_life=None):
    alpha_names = _alpha_names(wide)
    rows: list[dict[str, object]] = []
    for period in _prediction_periods(wide):
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
    return _gls_prediction_from_records(wide, records, window=window,
        all_periods=_prediction_periods(wide))


def _gls_prediction_from_records(wide, records, *, window, all_periods):
    alpha_names = _alpha_names(wide)
    rows: list[dict[str, object]] = []
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
    return _ols_prediction_from_records(wide, records, window=window,
        factor_names=factor_names, all_periods=_prediction_periods(wide),
        period_diagnostics=period_diagnostics)


def _ols_prediction_from_records(wide, records, *, window, factor_names, all_periods,
                                 period_diagnostics):
    alpha_names = _alpha_names(wide)
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
    for period in _prediction_periods(wide):
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


def _available_history(records, period: date, window: int, all_periods: list[date], *, positions=None):
    eligible = [
        record
        for record in records
        if record[0] < period and record[1] <= period.toordinal()
    ]
    if len(eligible) < window:
        return None
    selected = eligible[-window:]
    positions = {value: index for index, value in enumerate(all_periods)} if positions is None else positions
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


def _prediction_period_work(state, calendar, changes):
    """Select new/revised samples and retained samples that just matured."""
    periods = set(changes)
    if state is None:
        return tuple(calendar)
    periods.update(day for day in calendar if str(day) not in state["records"])
    last = calendar[-1].toordinal()
    periods.update(date.fromisoformat(day) for day, record in state["records"].items()
        if record["status"] == "pending" and record["available_at"] is not None
        and record["available_at"] <= last and date.fromisoformat(day) in calendar)
    return tuple(day for day in calendar if day in periods)


def _valid_prediction_state(state):
    """Reject missing original statistics, incomplete axes and malformed fits."""
    try:
        calendar = [date.fromisoformat(day) for day in state["calendar"]]
        count = state["alpha_count"]
        if (not calendar or calendar != sorted(set(calendar)) or state["through"] != str(calendar[-1])
                or isinstance(count, bool) or not isinstance(count, int) or count < 2
                or set(state["records"]) != set(state["calendar"])
                or set(state["fit_windows"]) != set(state["calendar"])):
            return False
        for day, record in state["records"].items():
            if (not isinstance(record, Mapping) or record["status"] not in {"pending", "fitted"}
                    or not isinstance(record["has_features"], bool)):
                return False
            available = record["available_at"]
            if available is not None and (isinstance(available, bool) or not isinstance(available, int)
                    or not 1 <= available <= date.max.toordinal()):
                return False
            statistics = record["statistics"]
            if record["status"] == "fitted":
                if available is None or not isinstance(record["input_identity"], str):
                    return False
                if isinstance(statistics, list):
                    if len(statistics) != count or any(value is not None and not np.isfinite(value) for value in statistics):
                        return False
                elif statistics is not None:
                    beta, covariance = np.asarray(statistics["beta"]), np.asarray(statistics["covariance"])
                    if (beta.shape != (count + 1,) or covariance.shape != (count + 1, count + 1)
                            or not np.isfinite(beta).all() or not np.isfinite(covariance).all()):
                        return False
            elif statistics is not None:
                return False
            history = state["fit_windows"][day]
            if history is not None and (not isinstance(history, list) or history != sorted(set(history))
                    or any(item not in state["records"] or item >= day
                        or state["records"][item]["status"] != "fitted"
                        or state["records"][item]["available_at"] > date.fromisoformat(day).toordinal()
                        for item in history)):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _saved_records(state):
    records = []
    for day, item in state["records"].items():
        if item["status"] != "fitted":
            continue
        statistics = item["statistics"]
        if isinstance(statistics, list):
            estimate = np.asarray(statistics, dtype=float)
        elif statistics is None:
            estimate = None
        else:
            estimate = (np.asarray(statistics["beta"], dtype=float),
                        np.asarray(statistics["covariance"], dtype=float))
        records.append((date.fromisoformat(day), item["available_at"], estimate))
    return sorted(records, key=lambda item: item[0])


def _update_prediction_statistics(operator, wide, targets, availability, state, calendar, work):
    data = _join_training(wide, targets, availability)
    alpha_names = _alpha_names(wide)
    old = deepcopy(state)
    records = {} if state is None else {day: deepcopy(record) for day, record in state["records"].items()
                                      if date.fromisoformat(day) in calendar}
    requested = tuple(calendar) if work is None else tuple(date.fromisoformat(day) for day in work["input_periods"])
    changed = set(requested)
    for period in requested:
        sample = data.filter(pl.col(TIME) == period)
        training = bool(_training_periods(sample))
        available_at = _period_availability(sample) if training else None
        record = {"has_features": bool(sample.height), "available_at": available_at,
                  "status": "pending", "statistics": None, "diagnostic": None,
                  "input_identity": None}
        if training and available_at <= calendar[-1].toordinal():
            if operator.kind in {"ols", "gls"}:
                fitted, diagnostic = _cross_sectional_regression_records(sample, alpha_names)
                estimate = fitted[0][2]
                record["statistics"] = (None if estimate is None else
                    {"beta": estimate[0].tolist(), "covariance": estimate[1].tolist()})
                record["diagnostic"] = diagnostic.to_dicts()[0]
            else:
                metric = "quantile_rank" if operator.kind == "quantile_ic_weighted" else "spearman"
                fitted = _rank_records(sample, alpha_names, metric=metric, quantiles=operator.quantiles)
                record["statistics"] = [float(value) if np.isfinite(value) else None for value in fitted[0][2]]
            # This training table has a fixed feature/target channel order.
            # Preserve its established checkpoint evidence policy.
            record.update(status="fitted", input_identity=hash_dataframe(sample, policy=DATAFRAME_HASH_V1))
        records[str(period)] = record
    result = {"through": str(calendar[-1]), "calendar": [str(day) for day in calendar],
              "alpha_count": len(alpha_names), "records": records, "fit_windows": {}}
    fitted = _saved_records(result)
    outputs = []
    influences = []
    if old is not None:
        influences.extend(date.fromisoformat(day) for day in set(old["calendar"]) ^ set(result["calendar"]))
        for period in requested:
            for record in (old["records"].get(str(period)), records[str(period)]):
                if record is not None and record["status"] == "fitted":
                    influences.append(max(period + timedelta(days=1), date.fromordinal(record["available_at"])))
    first_influence = min(influences) if influences else None
    positions = {day: index for index, day in enumerate(calendar)}
    changed_features = (set(calendar) if old is None else
        {date.fromisoformat(day) for day, roles in (work or {}).get("changes", {}).items() if "features" in roles})
    for period in calendar:
        if old is not None and str(period) in old["fit_windows"] and (first_influence is None or period < first_influence):
            selected = old["fit_windows"][str(period)]
        else:
            history = _available_history(fitted, period, operator.window, list(calendar), positions=positions)
            selected = None if history is None else [str(item[0]) for item in history]
        result["fit_windows"][str(period)] = selected
        original = None if old is None else old["fit_windows"].get(str(period))
        if (old is None or str(period) not in old["records"] or period in changed_features
                or original != selected or selected is not None and any(date.fromisoformat(day) in changed for day in selected)):
            outputs.append(str(period))
    result["repaired_output_periods"] = outputs
    return result


def _checkpoint_prediction(operator, wide, targets, availability):
    """Exchange per-period numerical evidence through Core's checkpoint runtime."""
    calendar = tuple(operator_execution_calendar()) or tuple(_prediction_periods(wide))
    if not calendar:
        return _rows_frame([])
    saved = restored_operator_state()
    if saved is not None and (not _valid_prediction_state(saved)
            or saved["alpha_count"] != len(_alpha_names(wide))):
        raise ValueError("Prediction needs original per-period statistical evidence maintenance")
    work = operator_period_work()
    if work is None or work["phase"] == "statistics":
        state = _update_prediction_statistics(operator, wide, targets, availability, saved, calendar, work)
        if work is not None:
            save_operator_state(state)
            return _rows_frame([])
    else:
        if saved is None:
            raise ValueError("Prediction phase requires admitted per-period statistics")
        state = deepcopy(saved)
    records = _saved_records(state)
    if operator.kind == "ols":
        result = _ols_prediction_from_records(wide, records, window=operator.window,
            factor_names=tuple(_alpha_names(wide)), all_periods=list(calendar),
            period_diagnostics=_fama_macbeth_period_frame([])).prediction
    elif operator.kind == "gls":
        result = _gls_prediction_from_records(wide, records, window=operator.window, all_periods=list(calendar))
    else:
        result = _rank_prediction_from_records(wide, records, window=operator.window,
            all_periods=list(calendar), half_life=operator.half_life)
    state.pop("repaired_output_periods", None)
    save_operator_state(state)
    return result


def _prediction_contract(operator_type: type[PredictionOperator]) -> OperationContract:
    if operator_type in {IdentityPredictionOperator, EqualWeightPredictionOperator}:
        return replace(_PREDICTION_CONTRACT, history=0)
    return replace(_PREDICTION_CONTRACT, checkpoint_replay=True,
        checkpoint_validator=_valid_prediction_state, checkpoint_periods=_prediction_period_work)


for _operator_type in (
    IdentityPredictionOperator,
    EqualWeightPredictionOperator,
    ICWeightedPredictionOperator,
    ICWeightedDecayPredictionOperator,
    QuantileICWeightedPredictionOperator,
    OLSPredictionOperator,
    GLSPredictionOperator,
):
    _name = "prediction:" + _operator_type.kind
    OPERATOR_REGISTRY.add(_name, Operator(_operator_type._compute_frames, registry_name=_name,
        contract=_prediction_contract(_operator_type),
        minimum_inputs=1, maximum_inputs=None,
        output_type="prediction", factory=_operator_type))


__all__ = [
    "EqualWeightPredictionOperator",
    "FamaMacBethOLSResult",
    "GLSPredictionOperator",
    "ICWeightedDecayPredictionOperator",
    "ICWeightedPredictionOperator",
    "IdentityPredictionOperator",
    "OLSPredictionOperator",
    "PredictionOperator",
    "PredictionTrainingContext",
    "QuantileICWeightedPredictionOperator",
    "fama_macbeth_ols_prediction",
    "quantile_rank_information_coefficient",
]
