from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .node import Node
from .panel import Panel

if TYPE_CHECKING:
    from .graph import Graph


def as_node(source: "Panel | Graph[Panel]", *, kind: str) -> Node:
    from .graph import Graph

    if isinstance(source, Panel):
        return source
    if isinstance(source, Graph):
        return source._single_output()
    expectation = "Panel or Graph inputs" if kind == "Composer" else "a Panel or Graph"
    raise TypeError(f"{kind} expects {expectation}")


def operation_name(operation: Callable[..., Any]) -> str:
    module = getattr(operation, "__module__", "")
    qualname = getattr(operation, "__qualname__", repr(operation))
    return f"{module}.{qualname}" if module else qualname
