"""Explicit numerical checkpoint exchange; applications own prefix proof and I/O."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_context = ContextVar("operator_checkpoints", default=None)
_node = ContextVar("operator_checkpoint_node", default=None)


@dataclass
class OperatorCheckpoints:
    restored: dict[str, Any]
    captured: dict[str, Any] = field(default_factory=dict)
    calendar: tuple = ()


@contextmanager
def capture_operator_checkpoints(restored=None, *, calendar=()):
    """Restore/capture generic state without reading files or account holdings.

    The caller must verify the complete causal input prefix before restoring.
    Operators validate their numerical signature independently.
    """
    context = OperatorCheckpoints(dict(restored or {}), calendar=tuple(calendar))
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


@contextmanager
def operator_checkpoint_node(name, signature):
    token = _node.set((name, signature))
    try:
        yield
    finally:
        _node.reset(token)


def restored_operator_state():
    context, node = _context.get(), _node.get()
    if context is None or node is None:
        return None
    saved = context.restored.get(node[0])
    if saved is not None and saved["signature"] != node[1]:
        raise ValueError("operator checkpoint numerical signature changed")
    return None if saved is None else saved["state"]


def operator_execution_calendar():
    """Explicit execution sessions, including sessions with no active assets."""
    context = _context.get()
    return () if context is None else context.calendar


def checkpoint_capture_enabled():
    return _context.get() is not None and _node.get() is not None


def save_operator_state(state):
    context, node = _context.get(), _node.get()
    if context is not None and node is not None:
        context.captured[node[0]] = {"signature": node[1], "state": state}
