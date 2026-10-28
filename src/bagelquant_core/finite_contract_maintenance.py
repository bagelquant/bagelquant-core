"""Explicit reference-only admission for previously undeclared finite built-ins.

Only adjacent change and bounded forward fill have this equivalence: their
whole-domain execution equals the existing finite policy without a numerical
reset. This is never an automatic update fallback or a generic policy upgrade.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json

from .execution import ExecutionRuntime
from .graph import Graph
from .graph_management import parents
from .hashing import hash_mapping
from .incremental import (CANONICAL_SESSIONS, FINITE_EXECUTION_POLICY,
    _canonical_halos, _reference_node, _relevant_parameters, finite_execution_policies, scoped_domain)
from .logical import LogicalGraphSpec
from .materialization import MaterializationKey, NodeMaterialization

MODE = "finite_contract.v1"
_OPERATORS = {
    "bagelquant_core.operator.streaks.diff_from_last_change",
    "bagelquant_core.operator.streaks.pct_change_from_last_change",
    "bagelquant_core.operator.missing.ffill",
}


def plan_maintenance(store, spec, identity, *, original_update, context_id,
        original_context, original_parents, original_projection,
        report_id, report_digest, check_canceled=lambda: None):
    """Authenticate old generic execution and register exact finite block proofs.

    External projections must come from the original immutable Source. Their
    content_token is the raw interval token used by Graph.plan_update input_proof;
    Core binds its actual Domain/type/traces. Stored parents use Core's own
    registered interval authority. No kernel or numerical value reader is used.
    Admission writes metadata and must be explicitly requested as maintenance.
    """
    from .proof_maintenance import ConsumedInputEvidence, PLAN_DDL
    check_canceled()
    if not report_id or len(report_digest) != 64 or any(c not in "0123456789abcdef" for c in report_digest):
        raise ValueError("Finite contract maintenance requires an immutable caller report")
    original = store.manifest(identity)
    if original is None:
        raise ValueError("Original finite contract maintenance authority is missing")
    key = MaterializationKey.from_dict(original["key"])
    declared = {node.node_id: node for node in spec.nodes}
    node = declared[key.node_id]
    if (node.operator not in _OPERATORS or original["execution_policy"] is not None
            or original["artifacts"] or original["checkpoint"] is not None or original["training_audits"]):
        raise ValueError("Finite contract maintenance requires a supported original stateless generic result")
    update = store.describe_update(original_update)
    if (update["results"].get(context_id, {}).get(node.node_id) != identity
            or not store.update_covers(original_update, context_id, node_ids=[node.node_id])):
        raise ValueError("Finite contract maintenance requires original completed update authority")
    if set(original_parents) != set(parents(node)):
        raise ValueError("Finite contract maintenance must bind every exact original parent")
    original_parents = {parent: original_parents[parent] for parent in parents(node)}
    parameters = original_context.get("parameters", {})
    policy = finite_execution_policies(spec, anchor=original_context["anchor"]).get(node.node_id)
    if policy is None:
        raise ValueError("Finite contract maintenance requires a declared finite history")
    local = LogicalGraphSpec({node.node_id: node.node_id}, spec.nodes)
    bound = Graph.from_logical_spec(local,
        inputs={declared[parent].input_key: value for parent, value in original_parents.items()
                if declared[parent].node_type == "input"},
        node_bindings={parent: value for parent, value in original_parents.items()
                       if declared[parent].node_type == "operator"}, parameter_bindings=parameters)
    planners = []
    for selected in (None, policy):
        runtime = ExecutionRuntime(node_contexts={node.node_id: hash_mapping({"history_policy": selected,
            "information_cutoff": original_context.get("information_cutoff")})}, _capture_contexts=True)
        keys = runtime.plan_materialization_keys(bound)
        planners.append((runtime, keys[node.node_id]))
    if planners[0][1] != key:
        raise ValueError("Original source/context bindings do not reproduce the completed generic key")
    domain = planners[0][0].node_domains[node.node_id]
    if (str(domain.times.min()) != policy["anchor"] or str(domain.times.max()) != original["through"]
            or store.domain_matches(identity, domain) is not True
            or any(not value.domain.times.equals(domain.times) for value in original_parents.values())):
        raise ValueError("Finite contract maintenance requires original indexed coordinates and complete calendar")
    source_authority = {}
    for parent, value in original_parents.items():
        if declared[parent].node_type == "operator":
            saved = store.describe(value.identity)
            if (saved is None or saved["key"]["node_id"] != parent
                    or saved["value_type"] != value.value_type.value
                    or saved["trace_columns"] != list(value.trace_columns)
                    or store.domain_matches(value.identity, value.domain) is not True):
                raise ValueError("Finite contract maintenance requires original registered parent authority")
            source_authority[parent] = {"identity": value.identity, "content_identity": saved["content_identity"]}
    calendar = domain.times.to_list()
    halos = _canonical_halos(domain, policy["history_observations"])
    publication_policy = {**policy, "parameter_bindings": _relevant_parameters(node, parameters)}
    blocks = []
    for begin in range(0, len(calendar), CANONICAL_SESSIONS):
        check_canceled()
        first, last = calendar[begin], calendar[min(begin + CANONICAL_SESSIONS, len(calendar)) - 1]
        warm = halos[begin]
        proofs, evidence = [], {}
        for parent, value in original_parents.items():
            own = scoped_domain(value.domain, warm, last)
            if parent in source_authority:
                token = store.interval_identity(value.identity, start=warm, end=last)
            else:
                proof = original_projection(parent, value, warm, last)
                if (not isinstance(proof, ConsumedInputEvidence) or proof.start != warm or proof.end != last
                        or proof.domain_signature != own.signature or proof.value_type != value.value_type.value
                        or proof.trace_columns != value.trace_columns):
                    raise ValueError("Finite contract maintenance requires exact original source projections")
                evidence[parent] = {**asdict(proof), "start": str(warm), "end": str(last)}
                token = value.source_interval_identity(proof.content_token, warm, last)
            if token is None:
                raise ValueError("Finite contract maintenance needs explicit original interval index maintenance")
            proofs.append(hash_mapping({"version": FINITE_EXECUTION_POLICY, "content": token,
                "domain": own.signature, "type": value.value_type.value, "traces": value.trace_columns}))
        output = scoped_domain(domain, first, last)
        block = MaterializationKey(node.node_id, key.implementation_id, tuple(proofs), output.signature,
            hash_mapping({"block": policy, "start": str(first), "end": str(last),
                "parameters": publication_policy["parameter_bindings"]}))
        blocks.append({"key": block.to_dict(), "start": str(first), "end": str(last),
            "read_start": str(warm), "sources": evidence})
    body = {"mode": MODE, "original_identity": identity, "original_content": original["content_identity"],
        "original_key": key.to_dict(), "original_update": original_update, "context_id": context_id,
        "original_context": original_context, "logical_node": node.to_dict(), "policy": publication_policy,
        "replacement_key": planners[1][1].to_dict(), "blocks": blocks, "parents": source_authority,
        "report_id": report_id, "report_digest": report_digest}
    body = json.loads(json.dumps(body, sort_keys=True, allow_nan=False))
    admission = {"identity": hash_mapping(body), **body}
    with store.transaction() as connection:
        check_canceled()
        connection.execute(PLAN_DDL)
        connection.execute("INSERT OR IGNORE INTO input_proof_rebinding_plans VALUES(?,?)",
            (admission["identity"], json.dumps(admission, sort_keys=True, separators=(",", ":"))))
        verify_maintenance(type(store), connection, admission, planners[1][1], publication_policy, require_blocks=False)
        check_canceled()
    return admission


def verify_maintenance(store_type, connection, reference, key, policy, *, visited=(), require_blocks=True,
        memo=None, dependencies=None):
    """Require immutable admission and all original numerical/update authority."""
    locator = ("finite_contract_plan", reference["identity"])
    admitted = None if memo is None else memo.get(locator, copy_value=False)
    if admitted is None:
        row = connection.execute("SELECT plan_json FROM input_proof_rebinding_plans WHERE identity=?",
            (reference["identity"],)).fetchone()
        if row is None:
            raise ValueError("Finite contract maintenance lacks immutable admission")
        plan = json.loads(row[0])
        body = {name: value for name, value in plan.items() if name != "identity"}
        if hash_mapping(body) != plan["identity"] or plan["identity"] != reference["identity"] or plan["mode"] != MODE:
            raise ValueError("Finite contract maintenance admission changed")
        keys = {MaterializationKey.from_dict(value).identity: value
            for value in [plan["replacement_key"], *(block["key"] for block in plan["blocks"])]}
        admitted = (plan, keys)
        if memo is not None:
            memo.put(locator, admitted, 2 * len(row[0].encode()))
    plan, keys = admitted
    handle = {name: plan[name] for name in ("mode", "identity", "original_identity")}
    if reference not in (handle, plan) or plan["policy"] != policy:
        raise ValueError("Finite contract maintenance admission changed")
    original_id = plan["original_identity"]
    if original_id in visited or original_id == key.identity:
        raise ValueError("Finite contract maintenance contains a reference cycle")
    if key.to_dict() != keys.get(key.identity) or key.node_id != plan["original_key"]["node_id"]:
        raise ValueError("Finite contract maintenance key differs from admitted original projections")
    authority = ("finite_contract_authority", plan["identity"])
    checked = None if memo is None else memo.get(authority, copy_value=False)
    if checked is not None:
        if {*visited, key.identity}.intersection(checked):
            raise ValueError("Finite contract maintenance contains a reference cycle")
    else:
        checked = _verify_original_authority(store_type, connection, plan, key, visited, memo)
        if memo is not None:
            memo.put(authority, checked, sum(len(identity) for identity in checked))
    if require_blocks and key.to_dict() == plan["replacement_key"]:
        block_ids = set(keys) - {key.identity}
        if set(visited).intersection(block_ids):
            raise ValueError("Finite contract maintenance contains a reference cycle")
        complete = ("finite_contract_blocks", plan["identity"])
        if memo is None or memo.get(complete, copy_value=False) is None:
            from .computation_index import read
            for block in plan["blocks"]:
                identity = MaterializationKey.from_dict(block["key"]).identity
                row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (identity,)).fetchone()
                if row is None:
                    raise ValueError("Finite contract maintenance lost a canonical block authority")
                saved = json.loads(row[0])
                store_type._verify_metadata(saved, identity)
                if read(connection, saved) is None:
                    raise ValueError("Finite contract maintenance lost a canonical block index authority")
            if memo is not None:
                memo.put(complete, True, 0)
        if dependencies is not None:
            dependencies.update(block_ids)
    if dependencies is not None:
        dependencies.update(checked)
    return plan


def _verify_original_authority(store_type, connection, plan, key, visited, memo):
    """Authenticate each shared admission ancestry once per bounded session."""
    original_id = plan["original_identity"]
    dependencies = {original_id}
    original = store_type._verified_reference_manifest(connection, original_id, plan["original_content"], memo=memo)
    from .computation_index import read
    if read(connection, original) is None:
        raise ValueError("Finite contract maintenance lost original result index authority")
    if (original["content_identity"] != plan["original_content"] or original["key"] != plan["original_key"]
            or original["execution_policy"] is not None or original.get("proof_rebinding") is not None):
        raise ValueError("Original finite contract maintenance authority changed")
    update = store_type._read_update_receipt(connection, plan["original_update"])
    if update["results"].get(plan["context_id"], {}).get(key.node_id) != original_id:
        raise ValueError("Finite contract maintenance lost original completed update authority")
    for parent in plan["parents"].values():
        saved = store_type._verified_reference_manifest(connection, parent["identity"], parent["content_identity"], memo=memo)
        if read(connection, saved) is None:
            raise ValueError("Finite contract maintenance lost original parent index authority")
        if saved["content_identity"] != parent["content_identity"]:
            raise ValueError("Original finite contract parent authority changed")
        dependencies.update(store_type._verify_reference_sources(connection, saved,
            visiting={*visited, key.identity, original_id}, memo=memo))
    return tuple(sorted(dependencies))


def apply_maintenance(store, plan, *, check_canceled=lambda: None):
    """Publish complete canonical certificates and finally their aggregate reference."""
    check_canceled()
    replacement = MaterializationKey.from_dict(plan["replacement_key"])
    with store.read_transaction() as connection:
        verify_maintenance(type(store), connection, plan, replacement, plan["policy"], require_blocks=False)
    original = store.lookup(MaterializationKey.from_dict(plan["original_key"]), evidence=False)
    entries = [*plan["blocks"], {"key": plan["replacement_key"],
        "start": str(original.panel.domain.times.min()), "end": str(original.panel.domain.times.max())}]
    reference = {name: plan[name] for name in ("mode", "identity", "original_identity")}
    for entry in entries:
        check_canceled()
        key = MaterializationKey.from_dict(entry["key"])
        if store.describe(key.identity) is not None:
            continue
        domain = original.panel.domain.window(date.fromisoformat(entry["start"]), date.fromisoformat(entry["end"]))
        value = _reference_node(original.panel.lazy(include_traces=True), domain, key,
            value_type=original.panel.value_type, trace_columns=original.panel.trace_columns)
        references = (tuple(MaterializationKey.from_dict(block["key"]).identity for block in plan["blocks"])
            if entry["key"] == plan["replacement_key"] else
            ({"identity": original.key.identity, "start": entry["start"], "end": entry["end"]},))
        store.publish(NodeMaterialization(key, value), execution_policy=plan["policy"],
            value_references=references,
            _proof_rebinding=reference)
    check_canceled()
    return {"outcome": "rebound", "identity": replacement.identity,
        "original_identity": original.key.identity, "computed_sessions": 0, "blocks": len(plan["blocks"])}


def maintained_prefixes(store, partials, completed, policy):
    """Admit only an explicit complete certificate for this retained completion."""
    originals = {saved["identity"]: saved for saved in completed}
    result = []
    for start in {saved["coverage_start"] for saved in completed}:
        for candidate in partials.get(start, ()):
            saved = store.manifest(candidate["identity"])
            reference = saved.get("proof_rebinding") or {}
            original = originals.get(reference.get("original_identity"))
            if (reference.get("mode") != MODE or original is None or saved["execution_policy"] != policy
                    or saved["through"] != original["through"]):
                continue
            with store.read_transaction() as connection:
                plan = verify_maintenance(type(store), connection, reference,
                    MaterializationKey.from_dict(saved["key"]), policy, memo=store._read_metadata_memo())
            # manifest() authenticated the aggregate and its complete original
            # block chain; do not traverse every alias a second time here.
            if saved["key"] == plan["replacement_key"]:
                result.append(store.describe(saved["identity"]))
    return result
