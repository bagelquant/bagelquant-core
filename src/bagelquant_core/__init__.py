"""Public API for BagelQuant Core graph and panel primitives.

Import from this module for the stable surface used by factor workflow code:
``Domain`` and ``Panel`` define aligned research data, ``Graph`` represents lazy
operations, and ``ExecutionRuntime`` evaluates graph outputs with memoization.
"""

from .execution import ExecutionRuntime
from .graph import CompiledGraph, Graph, GraphSpec, GraphValidationError
from .operation_contract import (
    ExecutionMode,
    InputDensity,
    OperationContract,
    TraceRule,
)
from .panel import CategoryPanel, Domain, Panel
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
    "InputDensity",
    "OperationContract",
    "Panel",
    "TraceRule",
    "pct_change_frame",
]
