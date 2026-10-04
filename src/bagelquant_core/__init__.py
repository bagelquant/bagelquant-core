"""Public API for BagelQuant Core graph and panel primitives.

Import from this module for the stable surface used by factor workflow code:
``Domain`` and ``Panel`` define aligned research data, ``Graph`` represents lazy
operations, and ``ExecutionRuntime`` evaluates graph outputs with memoization.
"""

from .node import Node
from .logical import LogicalNodeSpec, LogicalGraphSpec, canonicalize_graph
from .materialization import MaterializationKey, MaterializationStore, NodeMaterialization, MaterializationLookup, MaterializationStatus, materialization_trace_identity
from .operator import Operator, OperationNode, OPERATOR_REGISTRY
from .portfolio_values import PortfolioValue, rebalance, rebalance_value, top_n, equal_weight, regularized_weights, exposure_constrained_weights
from .execution import ExecutionRuntime
from .graph import CompiledGraph, Graph, GraphSpec, GraphValidationError
from .machine_learning import (
    ElasticNetCandidate,
    ElasticNetConfig,
    ElasticNetModel,
    ElasticNetSearchMode,
    ElasticNetPredictionComposer,
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
from .operation_contract import (
    ExecutionMode,
    InputDensity,
    OperationContract,
    TraceRule,
    causal_history_requirements,
)
from .panel import CategoryPanel, Domain, Panel, PredictionPanel
from .prediction import (
    EqualWeightPredictionComposer,
    FamaMacBethOLSResult,
    GLSPredictionComposer,
    ICWeightedDecayPredictionComposer,
    ICWeightedPredictionComposer,
    IdentityPredictionComposer,
    OLSPredictionComposer,
    PredictionComposer,
    PredictionTrainingContext,
    QuantileICWeightedPredictionComposer,
    fama_macbeth_ols_prediction,
    quantile_rank_information_coefficient,
)
from .transformer import pct_change_frame, canonicalize_values, project_domain
from .prediction_processing import (
    PredictionSmoothingConfig,
    PredictionSmoothingResult,
    PredictionSmoothingState,
    smooth_prediction,
)

from .operator_state import capture_operator_checkpoints
from .training_operators import rolling_elastic_net_prediction, rolling_lightgbm_prediction, capture_training_audits, date_balanced_training_keys

__all__ = [
    "LogicalNodeSpec", "LogicalGraphSpec", "canonicalize_graph",
    "MaterializationKey", "MaterializationStore", "NodeMaterialization",
    "MaterializationLookup", "MaterializationStatus",
    "materialization_trace_identity",
    "canonicalize_values", "project_domain",
    "PortfolioValue", "rebalance", "rebalance_value", "top_n", "equal_weight", "regularized_weights",
    "Node", "Operator", "OperationNode", "OPERATOR_REGISTRY",
    "CategoryPanel",
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
    "ElasticNetPredictionComposer",
    "InputDensity",
    "LabelBoundary",
    "OperationContract",
    "Panel",
    "PredictionComposer",
    "PredictionPanel",
    "PredictionSmoothingConfig",
    "PredictionSmoothingResult",
    "PredictionSmoothingState",
    "PredictionTrainingContext",
    "QuantileICWeightedPredictionComposer",
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
    "IdentityPredictionComposer",
    "EqualWeightPredictionComposer",
    "ICWeightedDecayPredictionComposer",
    "ICWeightedPredictionComposer",
    "OLSPredictionComposer",
    "GLSPredictionComposer",
    "TraceRule",
    "pct_change_frame",
    "quantile_rank_information_coefficient",
    "smooth_prediction",
    "zero_preserving_rms_scaler_from_moments",
]

__all__ += ['rolling_elastic_net_prediction', 'rolling_lightgbm_prediction', 'capture_training_audits', 'date_balanced_training_keys']
__all__ += ['exposure_constrained_weights']
__all__ += ['capture_operator_checkpoints']
