"""Public API for BagelQuant Core graph and panel primitives.

Import from this module for the stable surface used by factor workflow code:
``Domain`` and ``Panel`` define aligned research data, ``Graph`` represents lazy
operations, and ``ExecutionRuntime`` evaluates graph outputs with memoization.
"""

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
    ZeroPreservingRmsScaler,
    build_expanding_walk_forward,
    elastic_net_alpha_max,
    elastic_net_candidates,
    equal_period_sample_weights,
    fit_elastic_net,
)
from .operation_contract import (
    ExecutionMode,
    InputDensity,
    OperationContract,
    TraceRule,
)
from .panel import CategoryPanel, Domain, Panel, PredictionPanel
from .prediction import (
    EqualWeightPredictionComposer,
    GLSPredictionComposer,
    ICWeightedPredictionComposer,
    IdentityPredictionComposer,
    OLSPredictionComposer,
    PredictionComposer,
    PredictionTrainingContext,
)
from .transformer import pct_change_frame

__all__ = [
    "CategoryPanel",
    "CompiledGraph",
    "Domain",
    "ExecutionMode",
    "ExecutionRuntime",
    "Graph",
    "GraphSpec",
    "GraphValidationError",
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
    "PredictionTrainingContext",
    "WalkForwardConfig",
    "WalkForwardFold",
    "ZeroPreservingRmsScaler",
    "build_expanding_walk_forward",
    "elastic_net_alpha_max",
    "elastic_net_candidates",
    "equal_period_sample_weights",
    "fit_elastic_net",
    "IdentityPredictionComposer",
    "EqualWeightPredictionComposer",
    "ICWeightedPredictionComposer",
    "OLSPredictionComposer",
    "GLSPredictionComposer",
    "TraceRule",
    "pct_change_frame",
]
