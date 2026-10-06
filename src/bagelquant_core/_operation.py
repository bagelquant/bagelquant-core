"""Shared Node validation and stable operator names."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any
from bagelquant_core.node import Node


def as_node(source: Node, *, kind: str) -> Node:
    if not isinstance(source, Node):
        raise TypeError(f"{kind} expects Node inputs")
    return source


def operation_name(operation: Callable[..., Any]) -> str:
    module = getattr(operation, "__module__", "")
    qualname = getattr(operation, "__qualname__", repr(operation))
    return f"{module}.{qualname}" if module else qualname
