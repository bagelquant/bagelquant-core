"""Explicit source-bound input proof maintenance and retained-authority checks."""
from __future__ import annotations

from collections.abc import Mapping, Callable
from dataclasses import dataclass, asdict
from datetime import date
import json
from typing import Any

from .domain import Domain
from .node import Node, ValueType
from .logical import LogicalGraphSpec
from .hashing import hash_mapping
from .materialization import MaterializationKey, NodeMaterialization
from .graph_management import parents
from .incremental import _canonical_halos, _reference_node, _relevant_parameters, scoped_domain, FINITE_EXECUTION_POLICY, CANONICAL_SESSIONS


@dataclass(frozen=True, slots=True)
class ConsumedInputEvidence:
    """Caller-owned original projection authority, including every consumed channel.

    The source owner supplies the exact selected-content token and immutable
    source receipt/digest. Core validates scope, type, traces and original key;
    it does not import the source owner or substitute output equality for input
    equality. A scalar projection declares no trace columns.
    """
    source_receipt: str
    source_digest: str
    content_token: str
    start: date
    end: date
    domain_signature: str
    value_type: str
    trace_columns: tuple[str, ...] = ()

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in
                (self.source_receipt, self.source_digest, self.content_token, self.domain_signature, self.value_type)):
            raise ValueError("Consumed input evidence requires original source and typed content identities")
        if len(self.source_digest) != 64 or any(value not in "0123456789abcdef" for value in self.source_digest):
            raise ValueError("Consumed input evidence requires a SHA256 original source digest")
        ValueType(self.value_type)
        if (type(self.start) is not date or type(self.end) is not date or self.end < self.start
                or not isinstance(self.trace_columns, tuple)
                or any(not isinstance(value, str) or not value for value in self.trace_columns)
                or len(set(self.trace_columns)) != len(self.trace_columns)):
            raise ValueError("Consumed input evidence requires ordered bounds and immutable trace declarations")


PLAN_DDL = """CREATE TABLE IF NOT EXISTS input_proof_rebinding_plans(
 identity TEXT PRIMARY KEY, plan_json TEXT NOT NULL)"""


def verify_rebinding(store_type, connection, plan, key, policy, *, visited=(), require_admission=True,
        memo=None, dependencies=None):
    """Validate immutable admission and its complete retained authority chain."""
    if plan.get("mode") == "finite_contract.v1":
        from .finite_contract_maintenance import verify_maintenance
        return verify_maintenance(store_type, connection, plan, key, policy,
            visited=visited, memo=memo, dependencies=dependencies)
    from .logical import LogicalNodeSpec
    if plan.get("mode") in {"cutoff_normalization.v1", "cutoff_tail_repair.v1"}:
        from .cutoff_proof import verify_normalization
        return verify_normalization(store_type, connection, plan, key, policy,
            visited=visited, require_admission=require_admission, memo=memo, dependencies=dependencies)
    try:
        body = {name: value for name, value in plan.items() if name != "identity"}
        if plan.get("identity") != hash_mapping(body):
            raise ValueError("Proof rebinding authority checksum differs")
        if require_admission:
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='input_proof_rebinding_plans'").fetchone()
            row = None if exists is None else connection.execute(
                "SELECT plan_json FROM input_proof_rebinding_plans WHERE identity=?", (plan["identity"],)).fetchone()
            if row is None or json.loads(row[0]) != json.loads(json.dumps(plan)):
                raise ValueError("Proof rebinding plan lacks immutable source admission; prepare it again")
        original_id = plan["original_identity"]
        if original_id in visited or key.identity == original_id:
            raise ValueError("Proof rebinding authority contains a cycle")
        if dependencies is not None:
            dependencies.add(original_id)
        row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (original_id,)).fetchone()
        if row is None:
            raise ValueError("Original proof rebinding authority is missing")
        original = json.loads(row[0])
        store_type._verify_metadata(original, original_id)
        if original.get("proof_rebinding") is not None:
            verify_rebinding(store_type, connection, original["proof_rebinding"],
                MaterializationKey.from_dict(original["key"]), original["execution_policy"],
                visited=(*visited, original_id), memo=memo, dependencies=dependencies)
        original_policy = original["execution_policy"]
        numerical_policy = {name: value for name, value in original_policy.items() if name != "parameter_bindings"}
        if (original["content_identity"] != plan["original_content"] or original["key"] != plan["original_key"]
                or original_policy != plan["original_policy"] or plan["policy"] != policy
                or numerical_policy != {name: value for name, value in policy.items() if name != "parameter_bindings"}
                or policy.get("name") != FINITE_EXECUTION_POLICY
                or original["artifacts"] or original["checkpoint"] is not None or original["training_audits"]):
            raise ValueError("Original proof rebinding authority changed")
        replacement = plan["replacement_key"]
        declared = plan["logical_node"]
        node = LogicalNodeSpec.create(**{name: declared[name] for name in
            ("node_type", "input_key", "operator", "parameters", "inputs", "panel_parameters")})
        bindings = _relevant_parameters(node, plan["parameters"])
        digest = plan["report_digest"]
        if (node.node_id != key.node_id or node.to_dict() != declared or node.node_type != "operator"
                or policy.get("parameter_bindings") != bindings
                or ("parameter_bindings" in original_policy and original_policy["parameter_bindings"] != bindings)
                or not isinstance(plan["report_id"], str) or not plan["report_id"]
                or not isinstance(digest, str) or len(digest) != 64
                or any(value not in "0123456789abcdef" for value in digest)):
            raise ValueError("Proof rebinding lacks the original logical/parameter/report contract")
        contexts = {hash_mapping({"block": numerical_policy, "start": original["coverage_start"],
            "end": original["through"], "parameters": value}) for value in (bindings, plan["parameters"])}
        if original["key"]["context_identity"] not in contexts:
            raise ValueError("Original parameter evidence does not reproduce the completed context")
        if (replacement != key.to_dict() or any(replacement[field] != original["key"][field]
                for field in ("node_id", "implementation_id", "domain_identity", "context_identity"))):
            raise ValueError("Proof rebinding changed the original numerical contract")
        ordered, evidence = plan["parents"], plan["evidence"]
        if (tuple(ordered) != parents(node) or len(set(ordered)) != len(ordered) or set(ordered) != set(evidence)):
            raise ValueError("Proof rebinding does not preserve every original parent")
        old_proofs, new_proofs = [], []
        for parent in ordered:
            value = evidence[parent]
            projections = []
            for label in ("original", "replacement"):
                raw = value[label]
                projection = ConsumedInputEvidence(**{**raw, "start": date.fromisoformat(raw["start"]),
                    "end": date.fromisoformat(raw["end"]), "trace_columns": tuple(raw["trace_columns"])})
                if (str(projection.start) != plan["read_start"] or str(projection.end) != original["through"]):
                    raise ValueError("Consumed projection lacks the exact original scope/type/trace authority")
                token = value[f"{label}_token"]
                if (label == "replacement" and (not isinstance(token, str) or not token)
                        or token is not None and not isinstance(token, str)):
                    raise ValueError("Replacement interval proof is unknown")
                content = token or {"source": value[f"{label}_source_identity"],
                    "start": str(projection.start), "end": str(projection.end)}
                derived = hash_mapping({"version": FINITE_EXECUTION_POLICY, "content": content,
                    "domain": projection.domain_signature, "type": projection.value_type, "traces": projection.trace_columns})
                if derived != value[f"{label}_proof"]:
                    raise ValueError("Proof rebinding interval token does not reproduce its declared parent proof")
                (old_proofs if label == "original" else new_proofs).append(derived)
                projections.append(projection)
            before, after = projections
            if any(getattr(before, field) != getattr(after, field) for field in (
                    "content_token", "start", "end", "domain_signature", "value_type", "trace_columns")):
                raise ValueError("Proof rebinding consumed input evidence changed")
        if old_proofs != original["key"]["inputs"] or new_proofs != replacement["inputs"]:
            raise ValueError("Proof rebinding does not preserve every original parent")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Proof rebinding has incomplete source authority") from error


def plan_rebinding(store, spec: LogicalGraphSpec, identity: str, *, domain: Domain,
                   original_parents: Mapping[str, Node], replacement_parents: Mapping[str, Node],
                   original_input_proof: Callable, replacement_input_proof: Callable,
                   original_projection: Callable, replacement_projection: Callable,
                   report_id: str, report_digest: str, parameters: Mapping[str, Any],
                   check_canceled: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Plan one exact canonical finite block from original source projections.

    All callbacks are explicit source-owner contracts. Proof callbacks produce
    the same fully bound interval tokens as the graph's input_proof reader;
    projection callbacks return ConsumedInputEvidence from frozen originals.
    Missing original bindings cannot be inferred from current values. This
    explicit maintenance step registers an immutable admission record after
    validating source callbacks; it never alters original numerical receipts.
    """
    check_canceled()
    if not report_id or len(report_digest) != 64 or any(value not in "0123456789abcdef" for value in report_digest):
        raise ValueError("Proof maintenance requires a content-addressed caller report")
    manifest = store.manifest(identity)
    if manifest is None:
        raise ValueError("Original proof maintenance receipt is missing")
    policy = manifest.get("execution_policy") or {}
    if policy.get("name") != FINITE_EXECUTION_POLICY:
        raise ValueError("Proof maintenance requires the original finite policy")
    if manifest["artifacts"] or manifest["checkpoint"] is not None or manifest["training_audits"]:
        raise ValueError("Finite proof maintenance cannot discard original operator evidence")
    original_key = MaterializationKey.from_dict(manifest["key"])
    declared = {node.node_id: node for node in spec.nodes}
    node = declared.get(original_key.node_id)
    if node is None or node.node_type != "operator":
        raise ValueError("Original proof receipt does not belong to the declared operator")
    numerical_policy = {name: value for name, value in policy.items() if name != "parameter_bindings"}
    bindings = _relevant_parameters(node, parameters)
    if "parameter_bindings" in policy and policy["parameter_bindings"] != bindings:
        raise ValueError("Original numerical parameter bindings changed")
    contexts = {hash_mapping({"block": numerical_policy, "start": manifest["coverage_start"],
        "end": manifest["through"], "parameters": value}) for value in (bindings, dict(parameters))}
    if original_key.context_identity not in contexts:
        raise ValueError("Original parameter evidence does not reproduce the completed context")
    replacement_policy = {**policy, "parameter_bindings": bindings}
    ordered = parents(node)
    if set(original_parents) != set(ordered) or set(replacement_parents) != set(ordered):
        raise ValueError("Proof maintenance must bind every original and replacement direct parent")
    calendar = domain.times.to_list()
    first, last = date.fromisoformat(manifest["coverage_start"]), date.fromisoformat(manifest["through"])
    if first not in calendar or last not in calendar:
        raise ValueError("Original completed block is outside the admitted calendar")
    begin = calendar.index(first)
    if begin % CANONICAL_SESSIONS or calendar.index(last) >= begin + CANONICAL_SESSIONS:
        raise ValueError("Proof maintenance requires an original canonical block, not an aggregate receipt")
    if str(calendar[0]) != policy["anchor"]:
        raise ValueError("Original canonical anchor changed")
    own = scoped_domain(domain, first, last)
    if own.signature != original_key.domain_identity:
        raise ValueError("Original output calendar or membership changed")
    warm = _canonical_halos(domain, policy["history_observations"])[begin]
    old_proofs, new_proofs, evidence = [], [], {}
    for parent in ordered:
        check_canceled()
        before, after = original_parents[parent], replacement_parents[parent]
        before_domain, after_domain = scoped_domain(before.domain, warm, last), scoped_domain(after.domain, warm, last)
        if (before_domain.signature != after_domain.signature or before.value_type != after.value_type
                or before.trace_columns != after.trace_columns):
            raise ValueError("Consumed parent calendar, membership, type or traces changed")
        original = original_projection(parent, before, warm, last)
        replacement = replacement_projection(parent, after, warm, last)
        for projection in (original, replacement):
            if (not isinstance(projection, ConsumedInputEvidence) or projection.start != warm or projection.end != last
                    or projection.domain_signature != before_domain.signature
                    or projection.value_type != before.value_type.value or projection.trace_columns != before.trace_columns):
                raise ValueError("Consumed projection lacks the exact original scope/type/trace authority")
        if original.content_token != replacement.content_token:
            raise ValueError("Consumed historical inputs changed; proof format maintenance cannot repair numerical values")
        old_token = original_input_proof(parent, before, warm, last)
        new_token = replacement_input_proof(parent, after, warm, last)
        if not isinstance(new_token, str) or not new_token:
            raise ValueError("Replacement interval proof is unknown")
        def bound(token, value):
            return hash_mapping({"version": FINITE_EXECUTION_POLICY,
                "content": token or {"source": value.identity, "start": str(warm), "end": str(last)},
                "domain": before_domain.signature, "type": value.value_type.value, "traces": value.trace_columns})
        old_proofs.append(bound(old_token, before))
        new_proofs.append(bound(new_token, after))
        def describe(value):
            return {**asdict(value), "start": str(value.start), "end": str(value.end)}
        evidence[parent] = {"original": describe(original), "replacement": describe(replacement),
            "original_proof": old_proofs[-1], "replacement_proof": new_proofs[-1],
            "original_token": old_token, "replacement_token": new_token,
            "original_source_identity": before.identity, "replacement_source_identity": after.identity}
    if tuple(old_proofs) != original_key.inputs:
        raise ValueError("Original source bindings do not reproduce the immutable completed key")
    replacement = MaterializationKey(original_key.node_id, original_key.implementation_id,
        tuple(new_proofs), original_key.domain_identity, original_key.context_identity)
    body = {"original_identity": identity, "original_content": manifest["content_identity"],
        "original_key": original_key.to_dict(), "replacement_key": replacement.to_dict(),
        "original_policy": policy, "policy": replacement_policy, "parameters": dict(parameters), "logical_node": node.to_dict(),
        "parents": ordered, "evidence": evidence, "report_id": report_id, "report_digest": report_digest,
        "read_start": str(warm)}
    plan = {"identity": hash_mapping(body), **body}
    check_canceled()
    with store.transaction() as connection:
        check_canceled()
        verify_rebinding(type(store), connection, plan, replacement, replacement_policy, require_admission=False)
        connection.execute(PLAN_DDL)
        encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
        existing = connection.execute("SELECT plan_json FROM input_proof_rebinding_plans WHERE identity=?", (plan["identity"],)).fetchone()
        if existing is not None and existing[0] != encoded:
            raise ValueError("Conflicting immutable proof maintenance admission")
        connection.execute("INSERT OR IGNORE INTO input_proof_rebinding_plans VALUES(?,?)", (plan["identity"], encoded))
        check_canceled()
    return plan


def apply_rebinding(store, plan: Mapping[str, Any], *, check_canceled=lambda: None) -> dict[str, Any]:
    """Publish only proven original references; never run operators or load values."""
    if plan.get("mode") == "cutoff_normalization.v1":
        from .cutoff_proof import apply_normalization
        return apply_normalization(store, plan, check_canceled=check_canceled)
    body = {name: value for name, value in plan.items() if name != "identity"}
    if plan.get("identity") != hash_mapping(body):
        raise ValueError("Proof maintenance plan identity differs")
    check_canceled()
    original = store.manifest(plan["original_identity"])
    if original is None or original["content_identity"] != plan["original_content"]:
        raise ValueError("Original proof maintenance authority changed")
    key = MaterializationKey.from_dict(plan["replacement_key"])
    with store.read_transaction() as connection:
        verify_rebinding(type(store), connection, plan, key, plan["policy"])
    if store.describe(key.identity) is not None:
        return {"identity": key.identity, "outcome": "up_to_date", "original_identity": original["identity"]}
    source = store.lookup(MaterializationKey.from_dict(original["key"]), evidence=False)
    value = _reference_node(source.panel.lazy(include_traces=True), source.panel.domain, key,
        value_type=source.panel.value_type, trace_columns=source.panel.trace_columns)
    check_canceled()
    store.publish(NodeMaterialization(key, value, artifacts=source.artifacts, checkpoint=source.checkpoint,
        training_audits=source.training_audits), execution_policy=plan["policy"],
        value_references=(source.key.identity,), _proof_rebinding=dict(plan))
    return {"identity": key.identity, "outcome": "rebound", "original_identity": original["identity"]}
