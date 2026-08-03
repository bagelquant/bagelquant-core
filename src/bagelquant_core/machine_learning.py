"""Deterministic machine-learning primitives for signal composition.

This module intentionally contains no investment-domain policy and no
backtesting dependency.  It owns the reusable contracts needed by an
application orchestrator: monthly expanding folds, zero-preserving feature
scaling, and a small weighted Elastic Net implementation whose intercept does
not require centering the persisted feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from math import ceil
from typing import Iterable, Mapping, Sequence

import numpy as np


class ElasticNetSearchMode(StrEnum):
    """Supported regularization-search modes."""

    RELATIVE_ALPHA_PATH = "relative_alpha_path"
    ABSOLUTE_ALPHA_VALUES = "absolute_alpha_values"


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Monthly expanding walk-forward window configuration."""

    initial_sample_months: int = 72
    validation_months: int = 24
    oos_months: int = 12
    refit_on_sample_and_validation: bool = True
    embargo_months: int = 0
    minimum_valid_validation_months: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "initial_sample_months",
            "validation_months",
            "oos_months",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.embargo_months, int)
            or isinstance(self.embargo_months, bool)
            or self.embargo_months < 0
        ):
            raise ValueError("embargo_months must be a non-negative integer")
        if self.embargo_months >= self.initial_sample_months:
            raise ValueError("embargo_months must be smaller than the sample window")
        minimum = ceil(0.75 * self.validation_months)
        configured = self.minimum_valid_validation_months
        if configured is not None:
            if not isinstance(configured, int) or isinstance(configured, bool):
                raise ValueError(
                    "minimum_valid_validation_months must be an integer"
                )
            if configured < minimum:
                raise ValueError(
                    "minimum_valid_validation_months cannot be below 75% of "
                    "validation_months"
                )
            if configured > self.validation_months:
                raise ValueError(
                    "minimum_valid_validation_months cannot exceed validation_months"
                )

    @property
    def effective_minimum_valid_validation_months(self) -> int:
        """Return the explicit threshold or the mandatory 75% floor."""

        return self.minimum_valid_validation_months or ceil(
            0.75 * self.validation_months
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "frequency": "monthly",
            "window_mode": "expanding",
            "initial_sample_months": self.initial_sample_months,
            "validation_months": self.validation_months,
            "oos_months": self.oos_months,
            "refit_on_sample_and_validation": self.refit_on_sample_and_validation,
            "embargo_months": self.embargo_months,
            "minimum_valid_validation_months": (
                self.effective_minimum_valid_validation_months
            ),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One immutable expanding sample/validation/OOS partition."""

    index: int
    sample_periods: tuple[date, ...]
    validation_periods: tuple[date, ...]
    oos_periods: tuple[date, ...]
    embargoed_periods: tuple[date, ...] = ()

    @property
    def training_periods(self) -> tuple[date, ...]:
        """Sample periods that remain after the embargo."""

        if not self.embargoed_periods:
            return self.sample_periods
        return self.sample_periods[: -len(self.embargoed_periods)]


def build_expanding_walk_forward(
    periods: Iterable[date], config: WalkForwardConfig
) -> tuple[WalkForwardFold, ...]:
    """Partition unique monthly periods into deterministic expanding folds.

    A new fold starts after each complete OOS block.  Incomplete trailing OOS
    periods are deliberately excluded so published artifacts always cover the
    configured horizon.
    """

    ordered = tuple(sorted(set(periods)))
    required = (
        config.initial_sample_months
        + config.validation_months
        + config.oos_months
    )
    if len(ordered) < required:
        return ()
    folds: list[WalkForwardFold] = []
    sample_end = config.initial_sample_months
    index = 0
    while sample_end + config.validation_months + config.oos_months <= len(ordered):
        validation_end = sample_end + config.validation_months
        oos_end = validation_end + config.oos_months
        sample = ordered[:sample_end]
        embargoed = (
            sample[-config.embargo_months :] if config.embargo_months else ()
        )
        folds.append(
            WalkForwardFold(
                index=index,
                sample_periods=sample,
                validation_periods=ordered[sample_end:validation_end],
                oos_periods=ordered[validation_end:oos_end],
                embargoed_periods=embargoed,
            )
        )
        index += 1
        sample_end += config.oos_months
    return tuple(folds)


@dataclass(frozen=True, slots=True)
class LabelBoundary:
    """Explicit dates governing one supervised training record."""

    feature_date: date
    signal_date: date
    execution_date: date
    label_start_date: date
    label_end_date: date
    label_available_date: date
    model_fit_cutoff: date

    def validate(self) -> None:
        """Raise when the record violates the no-look-ahead contract."""

        if not self.feature_date <= self.signal_date < self.execution_date:
            raise ValueError(
                "label boundary requires feature_date <= signal_date < execution_date"
            )
        if self.label_start_date != self.execution_date:
            raise ValueError("label_start_date must equal execution_date")
        if self.label_end_date <= self.label_start_date:
            raise ValueError("label_end_date must be after label_start_date")
        if self.label_available_date >= self.model_fit_cutoff:
            raise ValueError(
                "label_available_date must be strictly before model_fit_cutoff"
            )


@dataclass(frozen=True, slots=True)
class ZeroPreservingRmsScaler:
    """Column scaler that never changes the semantic meaning of zero."""

    feature_names: tuple[str, ...]
    scales: tuple[float, ...]
    active_indices: tuple[int, ...]
    dropped_columns: Mapping[str, str]

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        feature_names: Sequence[str],
        sample_weight: np.ndarray | None = None,
    ) -> ZeroPreservingRmsScaler:
        """Fit weighted RMS scales without subtracting a column mean."""

        matrix = _validated_matrix(values)
        names = tuple(feature_names)
        if matrix.shape[1] != len(names):
            raise ValueError("feature_names length must match matrix columns")
        weights = _validated_weights(sample_weight, matrix.shape[0])
        denominator = float(weights.sum())
        raw_scales = np.sqrt((weights[:, None] * matrix * matrix).sum(axis=0) / denominator)
        active: list[int] = []
        scales: list[float] = []
        dropped: dict[str, str] = {}
        for index, (name, scale) in enumerate(zip(names, raw_scales, strict=True)):
            if not np.isfinite(scale):
                dropped[name] = "non_finite_rms"
            elif scale <= 0:
                dropped[name] = "zero_rms"
            else:
                active.append(index)
                scales.append(float(scale))
        return cls(names, tuple(scales), tuple(active), dropped)

    @property
    def active_feature_names(self) -> tuple[str, ...]:
        """Names retained by the scaler in model-column order."""

        return tuple(self.feature_names[index] for index in self.active_indices)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply sample-fitted scales while preserving every input zero."""

        matrix = _validated_matrix(values)
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("matrix columns do not match fitted scaler")
        if not self.active_indices:
            return np.empty((matrix.shape[0], 0), dtype=float)
        return matrix[:, self.active_indices] / np.asarray(self.scales)

    def restore_coefficients(self, scaled: np.ndarray) -> np.ndarray:
        """Map active scaled coefficients back to all original columns."""

        coefficients = np.asarray(scaled, dtype=float)
        if coefficients.shape != (len(self.active_indices),):
            raise ValueError("coefficient count does not match active columns")
        restored = np.zeros(len(self.feature_names), dtype=float)
        restored[list(self.active_indices)] = coefficients / np.asarray(self.scales)
        return restored

    def to_dict(self) -> dict[str, object]:
        """Return the persisted scaling lineage."""

        scale_by_name = dict(
            zip(self.active_feature_names, self.scales, strict=True)
        )
        return {
            "method": "weighted_rms_without_centering",
            "feature_names": list(self.feature_names),
            "active_feature_names": list(self.active_feature_names),
            "scales": scale_by_name,
            "dropped_columns": dict(self.dropped_columns),
        }


@dataclass(frozen=True, slots=True)
class ElasticNetConfig:
    """Serializable Elastic Net search and solver settings."""

    search_mode: ElasticNetSearchMode = ElasticNetSearchMode.RELATIVE_ALPHA_PATH
    alpha_ratios: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    l1_ratio_values: tuple[float, ...] = (0.1, 0.5, 0.9)
    alpha_values: tuple[float, ...] = ()
    max_iter: int = 10_000
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        mode = ElasticNetSearchMode(self.search_mode)
        object.__setattr__(self, "search_mode", mode)
        _validate_positive_values(self.l1_ratio_values, "l1_ratio_values", upper=1.0)
        if mode == ElasticNetSearchMode.RELATIVE_ALPHA_PATH:
            _validate_positive_values(self.alpha_ratios, "alpha_ratios", upper=1.0)
            if self.alpha_values:
                raise ValueError(
                    "alpha_values are only allowed in absolute_alpha_values mode"
                )
        else:
            _validate_positive_values(self.alpha_values, "alpha_values")
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "search_mode": self.search_mode.value,
            "alpha_ratios": list(self.alpha_ratios),
            "l1_ratio_values": list(self.l1_ratio_values),
            "alpha_values": list(self.alpha_values),
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class ElasticNetCandidate:
    """One regularization candidate with stable selection identity."""

    alpha: float
    l1_ratio: float
    alpha_ratio: float | None

    @property
    def selection_key(self) -> tuple[float, float, float]:
        """Stable tie-break key preferring stronger and then more-L1 penalty."""

        return (-self.alpha, -self.l1_ratio, -(self.alpha_ratio or 0.0))


@dataclass(frozen=True, slots=True)
class ElasticNetModel:
    """Fitted weighted Elastic Net model on scaled feature columns."""

    intercept: float
    coefficients: tuple[float, ...]
    alpha: float
    l1_ratio: float
    iterations: int
    converged: bool

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Predict from the active, already-scaled matrix."""

        matrix = _validated_matrix(values)
        coefficients = np.asarray(self.coefficients)
        if matrix.shape[1] != coefficients.size:
            raise ValueError("matrix columns do not match fitted model")
        return self.intercept + matrix @ coefficients


@dataclass(frozen=True, slots=True)
class ElasticNetSignalComposer:
    """Serializable Core facade for an application-owned ML state machine.

    Core owns configuration validation, scaling, and candidate fitting.  The
    application orchestrator owns fold lifecycle, while a backtesting package
    evaluates frozen predictions.  Keeping the complete configuration here
    makes the model contract reproducible without introducing a Core-to-BT
    dependency.
    """

    walk_forward: WalkForwardConfig
    coverage: Mapping[str, object]
    target: Mapping[str, object]
    elastic_net: ElasticNetConfig
    validation: Mapping[str, object]

    def __post_init__(self) -> None:
        required_coverage = {
            "minimum_all_market_observations",
            "minimum_applicable_coverage",
            "required_marker_unknown_policy",
        }
        missing_coverage = required_coverage - set(self.coverage)
        if missing_coverage:
            raise ValueError(
                f"coverage configuration is missing {sorted(missing_coverage)}"
            )
        required_target = {"uuid", "revision", "revision_hash", "definition"}
        missing_target = required_target - set(self.target)
        if missing_target:
            raise ValueError(
                f"target configuration is missing {sorted(missing_target)}"
            )
        if "objective" not in self.validation:
            raise ValueError("validation configuration requires objective")
        object.__setattr__(self, "coverage", _json_copy(self.coverage))
        object.__setattr__(self, "target", _json_copy(self.target))
        object.__setattr__(self, "validation", _json_copy(self.validation))

    def to_dict(self) -> dict[str, object]:
        """Serialize every model-affecting composer parameter."""

        return {
            "kind": "elastic_net",
            "walk_forward": self.walk_forward.to_dict(),
            "coverage": _json_copy(self.coverage),
            "target": _json_copy(self.target),
            "elastic_net": self.elastic_net.to_dict(),
            "validation": _json_copy(self.validation),
            "scaling": {
                "method": "weighted_rms_without_centering",
                "fit_scope": "sample_per_fold",
                "target_scaled": False,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ElasticNetSignalComposer:
        """Validate and restore a serialized composer configuration."""

        walk = dict(value["walk_forward"])
        walk.pop("frequency", None)
        walk.pop("window_mode", None)
        return cls(
            walk_forward=WalkForwardConfig(**walk),
            coverage=dict(value["coverage"]),
            target=dict(value["target"]),
            elastic_net=ElasticNetConfig(**dict(value["elastic_net"])),
            validation=dict(value["validation"]),
        )

    def candidates(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[ElasticNetCandidate, ...]:
        """Materialize this composer's fold-specific candidate path."""

        return elastic_net_candidates(
            self.elastic_net,
            values,
            target,
            sample_weight=sample_weight,
        )

    def fit(
        self,
        values: np.ndarray,
        target: np.ndarray,
        candidate: ElasticNetCandidate,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> ElasticNetModel:
        """Fit one candidate using fixed non-search solver settings."""

        return fit_elastic_net(
            values,
            target,
            alpha=candidate.alpha,
            l1_ratio=candidate.l1_ratio,
            sample_weight=sample_weight,
            max_iter=self.elastic_net.max_iter,
            tolerance=self.elastic_net.tolerance,
        )


def equal_period_sample_weights(periods: Sequence[object]) -> np.ndarray:
    """Give every period the same total observation weight."""

    labels = np.asarray(periods, dtype=object)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("periods must be a non-empty one-dimensional sequence")
    weights = np.empty(labels.size, dtype=float)
    for period in dict.fromkeys(labels.tolist()):
        mask = labels == period
        weights[mask] = 1.0 / int(mask.sum())
    return weights


def elastic_net_alpha_max(
    values: np.ndarray,
    target: np.ndarray,
    *,
    l1_ratio: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute the smallest alpha yielding all-zero penalized coefficients."""

    matrix, response, weights = _validated_training_data(
        values, target, sample_weight
    )
    if not 0 < l1_ratio <= 1:
        raise ValueError("l1_ratio must be in (0, 1]")
    centered_target = response - np.average(response, weights=weights)
    correlations = np.abs(matrix.T @ (weights * centered_target)) / weights.sum()
    result = float(correlations.max(initial=0.0) / l1_ratio)
    if not np.isfinite(result):
        raise ValueError("alpha_max is non-finite")
    return result


def elastic_net_candidates(
    config: ElasticNetConfig,
    values: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[ElasticNetCandidate, ...]:
    """Materialize a fold-specific deterministic candidate grid."""

    candidates: list[ElasticNetCandidate] = []
    for l1_ratio in config.l1_ratio_values:
        if config.search_mode == ElasticNetSearchMode.RELATIVE_ALPHA_PATH:
            maximum = elastic_net_alpha_max(
                values,
                target,
                l1_ratio=l1_ratio,
                sample_weight=sample_weight,
            )
            for ratio in config.alpha_ratios:
                candidates.append(
                    ElasticNetCandidate(maximum * ratio, l1_ratio, ratio)
                )
        else:
            candidates.extend(
                ElasticNetCandidate(alpha, l1_ratio, None)
                for alpha in config.alpha_values
            )
    return tuple(candidates)


def fit_elastic_net(
    values: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 10_000,
    tolerance: float = 1e-6,
) -> ElasticNetModel:
    """Fit Elastic Net by deterministic coordinate descent.

    The objective is ``weighted_mse / 2 + alpha*l1*|beta| +
    alpha*(1-l1)*||beta||²/2``.  The intercept is updated separately and is
    never penalized; the feature matrix is never centered.
    """

    matrix, response, weights = _validated_training_data(
        values, target, sample_weight
    )
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be non-negative and finite")
    if not np.isfinite(l1_ratio) or not 0 < l1_ratio <= 1:
        raise ValueError("l1_ratio must be in (0, 1]")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    weight_sum = float(weights.sum())
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    intercept = float(np.average(response, weights=weights))
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        previous = coefficients.copy()
        previous_intercept = intercept
        intercept = float(
            np.average(response - matrix @ coefficients, weights=weights)
        )
        residual = response - intercept - matrix @ coefficients
        for column_index in range(matrix.shape[1]):
            column = matrix[:, column_index]
            residual = residual + column * coefficients[column_index]
            correlation = float(np.dot(weights * column, residual) / weight_sum)
            denominator = float(
                np.dot(weights, column * column) / weight_sum
                + alpha * (1.0 - l1_ratio)
            )
            coefficient = (
                _soft_threshold(correlation, alpha * l1_ratio) / denominator
                if denominator > 0
                else 0.0
            )
            coefficients[column_index] = coefficient
            residual = residual - column * coefficient
        maximum_change = max(
            float(np.max(np.abs(coefficients - previous), initial=0.0)),
            abs(intercept - previous_intercept),
        )
        if maximum_change <= tolerance:
            converged = True
            break
    return ElasticNetModel(
        intercept=intercept,
        coefficients=tuple(float(value) for value in coefficients),
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        iterations=iteration,
        converged=converged,
    )


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _validated_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix must contain only finite values")
    return matrix


def _validated_weights(
    sample_weight: np.ndarray | None, row_count: int
) -> np.ndarray:
    weights = (
        np.ones(row_count, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weights.shape != (row_count,):
        raise ValueError("sample_weight must contain one value per row")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample_weight must be finite, non-negative, and non-zero")
    return weights


def _validated_training_data(
    values: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _validated_matrix(values)
    response = np.asarray(target, dtype=float)
    if response.shape != (matrix.shape[0],):
        raise ValueError("target must contain one value per feature row")
    if not np.isfinite(response).all():
        raise ValueError("target must contain only finite values")
    weights = _validated_weights(sample_weight, matrix.shape[0])
    return matrix, response, weights


def _validate_positive_values(
    values: Sequence[float], name: str, *, upper: float | None = None
) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for value in values:
        if not np.isfinite(value) or value <= 0 or (upper is not None and value > upper):
            qualifier = f" and <= {upper}" if upper is not None else ""
            raise ValueError(f"{name} values must be positive, finite{qualifier}")


def _json_copy(value: Mapping[str, object]) -> dict[str, object]:
    import json

    return json.loads(json.dumps(value, sort_keys=True, default=str))


__all__ = [
    "ElasticNetCandidate",
    "ElasticNetConfig",
    "ElasticNetModel",
    "ElasticNetSearchMode",
    "ElasticNetSignalComposer",
    "LabelBoundary",
    "WalkForwardConfig",
    "WalkForwardFold",
    "ZeroPreservingRmsScaler",
    "build_expanding_walk_forward",
    "elastic_net_alpha_max",
    "elastic_net_candidates",
    "equal_period_sample_weights",
    "fit_elastic_net",
]
