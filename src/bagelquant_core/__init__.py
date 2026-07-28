"""Public API for BagelQuant Core graph and panel primitives.

Import from this module for the stable surface used by factor workflow code:
``Domain`` and ``Panel`` define aligned research data, ``Graph`` represents lazy
operations, and ``ExecutionRuntime`` evaluates graph outputs with memoization.
"""

from .execution import ExecutionRuntime
from .graph import Graph, GraphSpec, GraphValidationError
from .panel import CategoryPanel, Domain, Panel
from .transformer import pct_change_frame

__all__ = [
    "CategoryPanel",
    "Domain",
    "ExecutionRuntime",
    "Graph",
    "GraphSpec",
    "GraphValidationError",
    "Panel",
    "pct_change_frame",
]
