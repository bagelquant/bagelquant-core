"""Public API for BagelQuant Core graph and panel primitives.

Import from this module for the stable surface used by factor workflow code:
``Domain`` and ``Node`` define aligned research data, ``Graph`` represents lazy
operations, and ``ExecutionRuntime`` evaluates graph outputs with memoization.
"""

from bagelquant_core.node import Node
from bagelquant_core.logical import LogicalNodeSpec, LogicalGraphSpec, canonicalize_graph
from bagelquant_core.materialization import MaterializationKey, MaterializationStore, NodeMaterialization, MaterializationLookup, MaterializationStatus, materialization_trace_identity
from bagelquant_core.operator import Operator, OPERATOR_REGISTRY
from bagelquant_core.operator.portfolio import PortfolioValue, rebalance, rebalance_value, top_n, equal_weight, regularized_weights, exposure_constrained_weights
from bagelquant_core.execution import ExecutionRuntime
from bagelquant_core.graph import CompiledGraph, Graph, GraphSpec, GraphValidationError
from bagelquant_core.machine_learning import (
    ElasticNetCandidate,
    ElasticNetConfig,
    ElasticNetModel,
    ElasticNetSearchMode,
    LabelBoundary,
    WalkForwardConfig,
    WalkForwardFold,
    WeightedRegressionMoments,
    ZeroPreservingRmsScaler,
    build_expanding_walk_forward,
    elastic_net_alpha_max,
    elastic_net_alpha_max_from_moments,
    elastic_net_candidates,
    elastic_net_candidates_from_moments,
    equal_period_sample_weights,
    fit_elastic_net,
    fit_elastic_net_from_moments,
    zero_preserving_rms_scaler_from_moments,
)
from bagelquant_core.operation_contract import (
    ExecutionMode,
    InputDensity,
    OperationContract,
    TraceRule,
    causal_history_requirements,
)
from bagelquant_core.node import Domain
from bagelquant_core.operator.prediction import (
    EqualWeightPredictionOperator,
    FamaMacBethOLSResult,
    GLSPredictionOperator,
    ICWeightedDecayPredictionOperator,
    ICWeightedPredictionOperator,
    IdentityPredictionOperator,
    OLSPredictionOperator,
    PredictionOperator,
    PredictionTrainingContext,
    QuantileICWeightedPredictionOperator,
    fama_macbeth_ols_prediction,
    quantile_rank_information_coefficient,
)
from bagelquant_core.operator import pct_change_frame, canonicalize_values, project_domain
from bagelquant_core.prediction_processing import (
    PredictionSmoothingConfig,
    PredictionSmoothingResult,
    PredictionSmoothingState,
    smooth_prediction,
)

from bagelquant_core.operator_state import capture_operator_checkpoints
from bagelquant_core.operator.training import rolling_elastic_net_prediction, rolling_lightgbm_prediction, capture_training_audits, date_balanced_training_keys

from .store import CoreStore, RevisionConflict
from .artifact_verification import ArtifactVerification
from .node import ValueType
from .graph_management import MergeResult

__all__ = [
    "LogicalNodeSpec", "LogicalGraphSpec", "canonicalize_graph",
    "MaterializationKey", "MaterializationStore", "NodeMaterialization",
    "MaterializationLookup", "MaterializationStatus",
    "materialization_trace_identity",
    "canonicalize_values", "project_domain",
    "PortfolioValue", "rebalance", "rebalance_value", "top_n", "equal_weight", "regularized_weights",
    "Node", "Operator", "OPERATOR_REGISTRY",
    "CompiledGraph",
    "Domain",
    "ExecutionMode",
    "ExecutionRuntime",
    "causal_history_requirements",
    "Graph",
    "GraphSpec",
    "GraphValidationError",
    "FamaMacBethOLSResult",
    "ElasticNetCandidate",
    "ElasticNetConfig",
    "ElasticNetModel",
    "ElasticNetSearchMode",
    "InputDensity",
    "LabelBoundary",
    "OperationContract",
    "PredictionOperator",
    "PredictionSmoothingConfig",
    "PredictionSmoothingResult",
    "PredictionSmoothingState",
    "PredictionTrainingContext",
    "QuantileICWeightedPredictionOperator",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WeightedRegressionMoments",
    "ZeroPreservingRmsScaler",
    "build_expanding_walk_forward",
    "elastic_net_alpha_max",
    "elastic_net_alpha_max_from_moments",
    "elastic_net_candidates",
    "elastic_net_candidates_from_moments",
    "equal_period_sample_weights",
    "fit_elastic_net",
    "fit_elastic_net_from_moments",
    "fama_macbeth_ols_prediction",
    "IdentityPredictionOperator",
    "EqualWeightPredictionOperator",
    "ICWeightedDecayPredictionOperator",
    "ICWeightedPredictionOperator",
    "OLSPredictionOperator",
    "GLSPredictionOperator",
    "TraceRule",
    "pct_change_frame",
    "quantile_rank_information_coefficient",
    "smooth_prediction",
    "zero_preserving_rms_scaler_from_moments",
]

__all__ += ['rolling_elastic_net_prediction', 'rolling_lightgbm_prediction', 'capture_training_audits', 'date_balanced_training_keys']
__all__ += ['exposure_constrained_weights']
__all__ += ['capture_operator_checkpoints']

__all__ += ["CoreStore", "RevisionConflict", "ValueType", "MergeResult", "ArtifactVerification"]
