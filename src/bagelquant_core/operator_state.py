"""Explicit numerical checkpoint exchange; Core update plans own prefix proof and I/O."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy

_context = ContextVar("operator_checkpoints", default=None)
_node = ContextVar("operator_checkpoint_node", default=None)
_evidence = ContextVar("operator_result_evidence", default=None)


def operator_checkpoint_signature(registered, config):
    from .logical import normalized_operator_config
    return {"operator": registered.registry_name, "version": registered.version,
            "parameters": normalized_operator_config(registered.registry_name, config)}


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
    result = {} if context is None else {
        "restored": context.restored.get(node_id),
        "calendar": [str(day) for day in context.calendar],
    }
    if context is not None and node_id in context.anchor_calendars:
        result["anchor_calendar"] = [str(day) for day in context.anchor_calendars[node_id]]
    if context is not None and node_id in context.period_work:
        result["period_work"] = context.period_work[node_id]
    return result


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
    anchor_calendars: dict[str, tuple] = field(default_factory=dict)
    period_work: dict[str, dict[str, Any]] = field(default_factory=dict)


@contextmanager
def capture_operator_checkpoints(restored=None, *, calendar=(), anchor_calendars=None, period_work=None):
    """Restore/capture generic state without reading files or account holdings.

    The caller must verify the complete causal input prefix before restoring.
    Operators validate their numerical signature independently.
    """
    context = OperatorCheckpoints(dict(restored or {}), calendar=tuple(calendar),
        anchor_calendars={key: tuple(days) for key, days in (anchor_calendars or {}).items()},
        period_work=deepcopy(dict(period_work or {})))
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


@contextmanager
def operator_checkpoint_node(name, signature, *, anchor_calendar=None, execution_calendar=()):
    current = _context.get()
    retained = None if current is None else current.anchor_calendars.get(name)
    if anchor_calendar is not None and retained is not None:
        actual = tuple(anchor_calendar)
        if not actual or tuple(day for day in retained if actual[0] <= day <= actual[-1]) != actual:
            raise ValueError("retained explicit calendar differs from the actual auxiliary Node axis")
    token = _node.set((name, signature))
    context = _context.get()
    context_token = None
    if context is None and (anchor_calendar is not None or execution_calendar):
        context = OperatorCheckpoints({}, calendar=tuple(execution_calendar))
        context_token = _context.set(context)
    previous = None if context is None else context.anchor_calendars.get(name)
    if context is not None and anchor_calendar is not None and previous is None:
        context.anchor_calendars[name] = tuple(anchor_calendar)
    try:
        yield
    finally:
        if context is not None and anchor_calendar is not None and previous is None:
            context.anchor_calendars.pop(name, None)
        if context_token is not None:
            _context.reset(context_token)
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


def operator_period_work():
    """Admitted independent-period work; numerical owners never choose I/O."""
    context, node = _context.get(), _node.get()
    return None if context is None or node is None else context.period_work.get(node[0])


def operator_anchor_calendar():
    """Declared auxiliary Domain axis retained before bounded input loading.

    The update planner validates this actual axis against execution coverage and
    proves its completed prefix before state restoration. Frame-only execution
    does not substitute another parent's Domain for an explicit calendar.
    """
    context, node = _context.get(), _node.get()
    return () if context is None or node is None else context.anchor_calendars.get(node[0], ())


def checkpoint_capture_enabled():
    return _context.get() is not None and _node.get() is not None


def save_operator_state(state):
    context, node = _context.get(), _node.get()
    evidence = _evidence.get()
    if evidence is not None and node is not None:
        evidence.checkpoints[node[0]] = {"signature": node[1], "state": deepcopy(state)}
    if context is not None and node is not None:
        context.captured[node[0]] = {"signature": node[1], "state": state}
