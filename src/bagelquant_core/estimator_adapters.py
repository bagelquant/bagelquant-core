"""Generic numerical estimators; training lifecycle belongs to the application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from bagelquant_core.machine_learning import (
    ElasticNetCandidate,
    ElasticNetConfig,
    ElasticNetModel,
    ElasticNetSearchMode,
    WeightedRegressionMoments,
    ZeroPreservingRmsScaler,
    elastic_net_alpha_max,
    elastic_net_alpha_max_from_moments,
    elastic_net_candidates,
    elastic_net_candidates_from_moments,
    fit_elastic_net,
    fit_elastic_net_from_moments,
    zero_preserving_rms_scaler_from_moments,
)
from bagelquant_core.resources import active_resource_limits


@dataclass(frozen=True, slots=True)
class FittedEstimator:
    """Serializable fitted estimator plus its feature scaler."""

    scaler: ZeroPreservingRmsScaler
    model: ElasticNetModel


@dataclass(frozen=True, slots=True)
class LightGBMCandidate:
    """One governed LightGBM hyperparameter candidate."""

    num_leaves: int
    learning_rate: float
    min_data_in_leaf: int
    lambda_l1: float
    lambda_l2: float
    feature_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "feature_fraction": self.feature_fraction,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LightGBMCandidate":
        return cls(
            num_leaves=int(value["num_leaves"]),
            learning_rate=float(value["learning_rate"]),
            min_data_in_leaf=int(value["min_data_in_leaf"]),
            lambda_l1=float(value["lambda_l1"]),
            lambda_l2=float(value["lambda_l2"]),
            feature_fraction=float(value["feature_fraction"]),
        )


@dataclass(frozen=True, slots=True)
class FittedLightGBM:
    """Serializable LightGBM booster and its stable feature order."""

    model: Any
    feature_names: tuple[str, ...]
    candidate: LightGBMCandidate


class EstimatorAdapter(Protocol):
    """Model-family boundary used by the multi-horizon state machine."""

    kind: str
    supports_moments: bool

    def candidates(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray,
    ) -> tuple[ElasticNetCandidate, ...]: ...

    def fit_candidate(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        feature_names: Sequence[str],
        sample_weight: np.ndarray,
        candidate: ElasticNetCandidate,
    ) -> FittedEstimator: ...

    def refit(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        feature_names: Sequence[str],
        sample_weight: np.ndarray,
        candidate: ElasticNetCandidate,
    ) -> FittedEstimator: ...

    def predict(self, fitted: FittedEstimator, values: np.ndarray) -> np.ndarray: ...

    def checkpoint(self, fitted: FittedEstimator) -> Mapping[str, Any]: ...

    def restore(self, checkpoint: Mapping[str, Any]) -> FittedEstimator: ...

    def diagnostics(self, fitted: FittedEstimator) -> Mapping[str, Any]: ...

    def candidate_sort_key(self, candidate: Any) -> tuple[object, ...]: ...

    def candidate_summary(self, candidate: Any) -> Mapping[str, Any]: ...

    def is_converged(self, fitted: Any) -> bool: ...


class ElasticNetEstimatorAdapter:
    """Adapter over Core's deterministic weighted ElasticNet primitives."""

    kind = "elastic_net"
    supports_moments = True

    def __init__(self, settings: Mapping[str, Any]) -> None:
        solver = settings["solver"]
        search = settings["search"]
        self.config = ElasticNetConfig(
            search_mode=(
                ElasticNetSearchMode.RELATIVE_ALPHA_PATH
                if search["mode"] == "relative_path"
                else ElasticNetSearchMode.ABSOLUTE_ALPHA_VALUES
            ),
            alpha_ratios=tuple(
                float(value) for value in solver["relative_alpha_ratios"]
            ),
            l1_ratio_values=tuple(float(value) for value in solver["l1_ratios"]),
            alpha_values=tuple(
                float(value) for value in solver["absolute_alpha_values"]
            ),
            max_iter=int(solver["max_iter"]),
            tolerance=float(solver["tolerance"]),
        )

    def candidates(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray,
    ) -> tuple[ElasticNetCandidate, ...]:
        return elastic_net_candidates(
            self.config,
            values,
            target,
            sample_weight=sample_weight,
        )

    def fit_candidate(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        feature_names: Sequence[str],
        sample_weight: np.ndarray,
        candidate: ElasticNetCandidate,
    ) -> FittedEstimator:
        scaler = ZeroPreservingRmsScaler.fit(
            values,
            feature_names=feature_names,
            sample_weight=sample_weight,
        )
        scaled = scaler.transform(values)
        if scaled.shape[1] == 0:
            raise RuntimeError("estimator has no active feature columns")
        model = fit_elastic_net(
            scaled,
            target,
            alpha=candidate.alpha,
            l1_ratio=candidate.l1_ratio,
            sample_weight=sample_weight,
            max_iter=self.config.max_iter,
            tolerance=self.config.tolerance,
        )
        return FittedEstimator(scaler, model)

    def candidates_from_moments(
        self,
        moments: WeightedRegressionMoments,
        *,
        feature_names: Sequence[str],
    ) -> tuple[ElasticNetCandidate, ...]:
        """Build a fold path without retaining its observation matrix."""

        scaler = zero_preserving_rms_scaler_from_moments(
            moments, feature_names=feature_names
        )
        return elastic_net_candidates_from_moments(self.config, moments, scaler)

    def fit_candidate_from_moments(
        self,
        moments: WeightedRegressionMoments,
        *,
        feature_names: Sequence[str],
        candidate: ElasticNetCandidate,
    ) -> FittedEstimator:
        """Fit one absolute candidate from mergeable weighted moments."""

        scaler = zero_preserving_rms_scaler_from_moments(
            moments, feature_names=feature_names
        )
        model = fit_elastic_net_from_moments(
            moments,
            scaler,
            alpha=candidate.alpha,
            l1_ratio=candidate.l1_ratio,
            max_iter=self.config.max_iter,
            tolerance=self.config.tolerance,
        )
        return FittedEstimator(scaler, model)

    def refit_from_moments(
        self,
        moments: WeightedRegressionMoments,
        *,
        feature_names: Sequence[str],
        candidate: ElasticNetCandidate,
    ) -> FittedEstimator:
        """Refit a selected candidate on a new moment snapshot."""

        scaler = zero_preserving_rms_scaler_from_moments(
            moments, feature_names=feature_names
        )
        alpha = candidate.alpha
        if self.config.search_mode == ElasticNetSearchMode.RELATIVE_ALPHA_PATH:
            alpha = elastic_net_alpha_max_from_moments(
                moments,
                scaler,
                l1_ratio=candidate.l1_ratio,
            ) * float(candidate.alpha_ratio)
        refit_candidate = ElasticNetCandidate(
            alpha=alpha,
            l1_ratio=candidate.l1_ratio,
            alpha_ratio=candidate.alpha_ratio,
        )
        return self.fit_candidate_from_moments(
            moments,
            feature_names=feature_names,
            candidate=refit_candidate,
        )

    def refit(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        feature_names: Sequence[str],
        sample_weight: np.ndarray,
        candidate: ElasticNetCandidate,
    ) -> FittedEstimator:
        scaler = ZeroPreservingRmsScaler.fit(
            values,
            feature_names=feature_names,
            sample_weight=sample_weight,
        )
        scaled = scaler.transform(values)
        if scaled.shape[1] == 0:
            raise RuntimeError("estimator refit has no active feature columns")
        alpha = candidate.alpha
        if self.config.search_mode == ElasticNetSearchMode.RELATIVE_ALPHA_PATH:
            alpha = elastic_net_alpha_max(
                scaled,
                target,
                l1_ratio=candidate.l1_ratio,
                sample_weight=sample_weight,
            ) * float(candidate.alpha_ratio)
        refit_candidate = ElasticNetCandidate(
            alpha=alpha,
            l1_ratio=candidate.l1_ratio,
            alpha_ratio=candidate.alpha_ratio,
        )
        return self.fit_candidate(
            values,
            target,
            feature_names=feature_names,
            sample_weight=sample_weight,
            candidate=refit_candidate,
        )

    def predict(self, fitted: FittedEstimator, values: np.ndarray) -> np.ndarray:
        return fitted.model.predict(fitted.scaler.transform(values))

    def checkpoint(self, fitted: FittedEstimator) -> Mapping[str, Any]:
        return {
            "kind": self.kind,
            "scaler": fitted.scaler.to_dict(),
            "model": fitted.model.to_dict(),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> FittedEstimator:
        if checkpoint.get("kind") != self.kind:
            raise ValueError("estimator checkpoint kind does not match adapter")
        return FittedEstimator(
            ZeroPreservingRmsScaler.from_dict(checkpoint["scaler"]),
            ElasticNetModel.from_dict(checkpoint["model"]),
        )

    def diagnostics(self, fitted: FittedEstimator) -> Mapping[str, Any]:
        return {
            "feature_order": list(fitted.scaler.feature_names),
            "active_features": list(fitted.scaler.active_feature_names),
            "scaled_coefficients": list(fitted.model.coefficients),
            "original_scale_coefficients": fitted.scaler.restore_coefficients(
                np.asarray(fitted.model.coefficients)
            ).tolist(),
            "intercept": fitted.model.intercept,
            "alpha": fitted.model.alpha,
            "l1_ratio": fitted.model.l1_ratio,
            "iterations": fitted.model.iterations,
            "converged": fitted.model.converged,
        }

    def candidate_sort_key(self, candidate: ElasticNetCandidate) -> tuple[object, ...]:
        return (
            candidate.alpha,
            candidate.l1_ratio,
            candidate.alpha_ratio or 0.0,
        )

    def candidate_summary(self, candidate: ElasticNetCandidate) -> Mapping[str, Any]:
        return {
            "alpha": candidate.alpha,
            "l1_ratio": candidate.l1_ratio,
            "alpha_ratio": candidate.alpha_ratio,
        }

    def is_converged(self, fitted: FittedEstimator) -> bool:
        return fitted.model.converged


class LightGBMEstimatorAdapter:
    """Memory-bounded adapter over LightGBM's histogram tree learner."""

    kind = "lightgbm"
    supports_moments = False

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.solver = dict(settings["solver"])
        self._candidates = tuple(
            LightGBMCandidate.from_mapping(value)
            for value in settings["search"]["candidates"]
        )
        if not self._candidates:
            raise ValueError("LightGBM requires at least one candidate")

    def candidates(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray,
    ) -> tuple[LightGBMCandidate, ...]:
        del values, target, sample_weight
        return self._candidates

    def fit_candidate(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        feature_names: Sequence[str],
        sample_weight: np.ndarray,
        candidate: LightGBMCandidate,
    ) -> FittedLightGBM:
        lgb = _lightgbm()
        matrix = np.asarray(values, dtype=np.float32, order="C")
        labels = np.asarray(target, dtype=np.float32)
        weights = np.asarray(sample_weight, dtype=np.float32)
        dataset = lgb.Dataset(
            matrix,
            label=labels,
            weight=weights,
            feature_name=list(feature_names),
            free_raw_data=True,
            params={"max_bin": int(self.solver["max_bin"])},
        )
        params = {
            "objective": "regression_l2",
            "metric": "l2",
            "boosting_type": "gbdt",
            "num_leaves": candidate.num_leaves,
            "learning_rate": candidate.learning_rate,
            "min_data_in_leaf": candidate.min_data_in_leaf,
            "lambda_l1": candidate.lambda_l1,
            "lambda_l2": candidate.lambda_l2,
            "feature_fraction": candidate.feature_fraction,
            "max_bin": int(self.solver["max_bin"]),
            "max_depth": int(self.solver["max_depth"]),
            "min_gain_to_split": float(self.solver["min_gain_to_split"]),
            "num_threads": active_resource_limits().lightgbm_threads,
            "histogram_pool_size": active_resource_limits().histogram_pool_mib,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "seed": int(self.solver["seed"]),
            "feature_fraction_seed": int(self.solver["seed"]),
            "data_random_seed": int(self.solver["seed"]),
        }
        model = lgb.train(
            params,
            dataset,
            num_boost_round=int(self.solver["num_boost_round"]),
        )
        model.free_dataset()
        return FittedLightGBM(model, tuple(feature_names), candidate)

    def refit(
        self,
        values: np.ndarray,
        target: np.ndarray,
        *,
        feature_names: Sequence[str],
        sample_weight: np.ndarray,
        candidate: LightGBMCandidate,
    ) -> FittedLightGBM:
        return self.fit_candidate(
            values,
            target,
            feature_names=feature_names,
            sample_weight=sample_weight,
            candidate=candidate,
        )

    def predict(self, fitted: FittedLightGBM, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32, order="C")
        return np.asarray(fitted.model.predict(matrix), dtype=np.float64)

    def checkpoint(self, fitted: FittedLightGBM) -> Mapping[str, Any]:
        return {
            "kind": self.kind,
            "feature_names": list(fitted.feature_names),
            "candidate": fitted.candidate.to_dict(),
            "model": fitted.model.model_to_string(
                num_iteration=fitted.model.current_iteration()
            ),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> FittedLightGBM:
        if checkpoint.get("kind") != self.kind:
            raise ValueError("estimator checkpoint kind does not match adapter")
        lgb = _lightgbm()
        return FittedLightGBM(
            lgb.Booster(model_str=str(checkpoint["model"])),
            tuple(str(value) for value in checkpoint["feature_names"]),
            LightGBMCandidate.from_mapping(checkpoint["candidate"]),
        )

    def diagnostics(self, fitted: FittedLightGBM) -> Mapping[str, Any]:
        return {
            "feature_order": list(fitted.feature_names),
            "feature_importance_gain": fitted.model.feature_importance(
                importance_type="gain"
            ).tolist(),
            "feature_importance_split": fitted.model.feature_importance(
                importance_type="split"
            ).tolist(),
            "tree_count": fitted.model.current_iteration(),
            "candidate": fitted.candidate.to_dict(),
        }

    def candidate_sort_key(self, candidate: LightGBMCandidate) -> tuple[object, ...]:
        return (
            candidate.num_leaves,
            candidate.learning_rate,
            candidate.min_data_in_leaf,
            candidate.lambda_l1,
            candidate.lambda_l2,
            candidate.feature_fraction,
        )

    def candidate_summary(self, candidate: LightGBMCandidate) -> Mapping[str, Any]:
        return candidate.to_dict()

    def is_converged(self, fitted: FittedLightGBM) -> bool:
        del fitted
        return True


def _lightgbm():
    try:
        import lightgbm
    except ImportError as error:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("LightGBM is not installed") from error
    return lightgbm


def estimator_adapter(
    kind: str,
    settings: Mapping[str, Any],
) -> EstimatorAdapter:
    """Resolve one validated estimator adapter by stable recipe kind."""

    if kind == ElasticNetEstimatorAdapter.kind:
        return ElasticNetEstimatorAdapter(settings)
    if kind == LightGBMEstimatorAdapter.kind:
        return LightGBMEstimatorAdapter(settings)
    raise ValueError(f"unsupported estimator kind: {kind}")


__all__ = [
    "ElasticNetEstimatorAdapter",
    "EstimatorAdapter",
    "FittedEstimator",
    "FittedLightGBM",
    "LightGBMCandidate",
    "LightGBMEstimatorAdapter",
    "estimator_adapter",
]
