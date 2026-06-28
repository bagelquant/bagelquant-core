"""Public API for BagelQuant Core graph and panel primitives.

Import from this module for the stable surface used by factor workflow code:
``Domain`` and ``Panel`` define aligned research data, ``Graph`` represents lazy
operations, and ``ExecutionRuntime`` evaluates graph outputs with memoization.
"""

from .execution import ExecutionRuntime
from .graph import Graph
from .panel import CategoryPanel, Domain, Panel

__all__ = [
    "CategoryPanel",
    "Domain",
    "ExecutionRuntime",
    "Graph",
    "Panel",
]
