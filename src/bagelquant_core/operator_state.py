"""Explicit numerical checkpoint exchange; Core update plans own prefix proof and I/O."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy

_context = ContextVar("operator_checkpoints", default=None)
_node = ContextVar("operator_checkpoint_node", default=None)
_evidence = ContextVar("operator_result_evidence", default=None)


@dataclass
class NodeEvidence:
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoints: dict[str, Any] = field(default_factory=dict)
    training_audits: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@contextmanager
def capture_node_evidence():
    """Observe numerical output channels for one Runtime execution."""
    evidence = NodeEvidence()
    token = _evidence.set(evidence)
    try:
        yield evidence
    finally:
        _evidence.reset(token)


def current_operator_checkpoints():
    return _context.get()


def operator_input_context(node_id):
    """Restored state and explicit calendar participate in execution identity."""
    context = _context.get()
    return {} if context is None else {
        "restored": context.restored.get(node_id),
        "calendar": [str(day) for day in context.calendar],
    }


def save_operator_artifact(name: str, value: Any) -> None:
    """Publish a generic named numerical output alongside the primary Node."""
    evidence, node = _evidence.get(), _node.get()
    if evidence is not None and node is not None:
        evidence.artifacts.setdefault(node[0], {})[name] = deepcopy(value)


def save_training_audit(audit, *, node_id=None):
    evidence, node = _evidence.get(), _node.get()
    key = node_id if node_id is not None else None if node is None else node[0]
    if evidence is not None and key is not None:
        evidence.training_audits.setdefault(key, []).append(deepcopy(audit))


def replay_node_evidence(node_id, *, artifacts, checkpoint, training_audits):
    """An exact cache hit returns the evidence of the saved numerical run."""
    evidence = _evidence.get()
    if evidence is not None:
        evidence.artifacts[node_id] = deepcopy(dict(artifacts))
        evidence.training_audits[node_id] = deepcopy(list(training_audits))
        if checkpoint is not None:
            evidence.checkpoints[node_id] = deepcopy(checkpoint)
    context = _context.get()
    if context is not None and checkpoint is not None:
        context.captured[node_id] = deepcopy(checkpoint)
    from bagelquant_core.operator.training import replay_training_audits
    replay_training_audits(training_audits)


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
    evidence = _evidence.get()
    if evidence is not None and node is not None:
        evidence.checkpoints[node[0]] = {"signature": node[1], "state": deepcopy(state)}
    if context is not None and node is not None:
        context.captured[node[0]] = {"signature": node[1], "state": state}
