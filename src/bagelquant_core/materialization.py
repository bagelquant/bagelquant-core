"""Generic immutable numerical results; applications own durable storage.

Logical definitions never contain these execution keys. Input identities must
be backed by immutable content receipts supplied by the caller, not filenames
or a mutable "latest" pointer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable
from enum import StrEnum

from .hashing import hash_mapping
from .panel import Panel


@dataclass(frozen=True, slots=True)
class MaterializationKey:
    """A node under one implementation, causal input state and numeric domain."""

    node_id: str
    implementation_id: str
    inputs: tuple[str, ...]
    domain_identity: str
    context_identity: str = ""

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in
               (self.node_id, self.implementation_id, self.domain_identity)):
            raise ValueError("materialization identities must be nonempty strings")
        if not isinstance(self.context_identity, str):
            raise TypeError("materialization context identity must be a string")
        if any(not isinstance(parent, str) or not parent for parent in self.inputs):
            raise ValueError("materialization inputs must contain nonempty identities")
        object.__setattr__(self, "inputs", tuple(self.inputs))

    @property
    def identity(self) -> str:
        return hash_mapping(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "implementation_id": self.implementation_id,
            "inputs": list(self.inputs),
            "domain_identity": self.domain_identity,
            "context_identity": self.context_identity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MaterializationKey:
        return cls(
            node_id=value["node_id"],
            implementation_id=value["implementation_id"],
            inputs=tuple(value["inputs"]),
            domain_identity=value["domain_identity"],
            context_identity=value.get("context_identity", ""),
        )


def materialization_trace_identity(key: MaterializationKey, trace_columns: tuple[str, ...]) -> str | None:
    """Canonical receipt trace token, independent of execution blocks and aliases."""
    return (hash_mapping({"materialization": key.identity, "trace_columns": list(trace_columns)})
            if trace_columns else None)


@dataclass(frozen=True, slots=True)
class NodeMaterialization:
    """A typed value with the evidence produced by that exact computation.

Artifacts are generic named outputs (for example a decisions table), rather
than application object metadata. Store implementations must retain Panel
type, sparse keys, Domain, traces, checkpoints and fit audits together.
"""

    key: MaterializationKey
    panel: Panel
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: Mapping[str, Any] | None = None
    training_audits: tuple[Mapping[str, Any], ...] = ()


class MaterializationStatus(StrEnum):
    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"


@dataclass(frozen=True, slots=True)
class MaterializationLookup:
    """Admission status; a partial candidate is never an exact numerical hit."""

    status: MaterializationStatus
    materialization: NodeMaterialization | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MaterializationStatus(self.status))
        if self.status == MaterializationStatus.HIT and self.materialization is None:
            raise ValueError("a materialization hit requires a saved value")
        if self.status == MaterializationStatus.MISS and self.materialization is not None:
            raise ValueError("a materialization miss cannot contain a saved value")


@runtime_checkable
class MaterializationStore(Protocol):
    """Storage adapter implemented by an application, never by Core.

Query verifies the immutable receipt and labels exact, partial or missing values. Publish
must atomically save all result channels or raise; it must not silently replace
another computation. Partial coverage and causal-prefix admission belong to
the caller's update planner, which submits the admitted execution block.
"""

    def query(self, key: MaterializationKey) -> MaterializationLookup: ...

    def publish(self, value: NodeMaterialization) -> None: ...


__all__ = ["MaterializationKey", "MaterializationStore", "NodeMaterialization",
           "MaterializationLookup", "MaterializationStatus", "materialization_trace_identity"]
