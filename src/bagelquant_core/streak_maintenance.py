"""Explicit checkpoint certification from original streak output and parent tails."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
import json

import polars as pl

from .computation_index import read as read_index, interval
from .execution import ExecutionRuntime
from .graph import Graph
from .graph_management import parents
from .hashing import hash_mapping
from .incremental import CANONICAL_SESSIONS, _reference_node, _relevant_parameters
from .logical import LogicalGraphSpec, LogicalNodeSpec
from .materialization import MaterializationKey, NodeMaterialization
from .operator import OPERATOR_REGISTRY
from .operator_state import operator_checkpoint_signature
from .resources import active_resource_limits
from .streak_state import STATE_SCHEMA, _number, valid_streak_state, validate_streak_calendar

MODE = "streak_checkpoint.v1"
_OPERATOR = "bagelquant_core.operator.streaks.streak_count"


def _replacement(original, checkpoint, policy):
    key = MaterializationKey.from_dict(original["key"])
    return replace(key, context_identity=hash_mapping({"mode": MODE,
        "original_identity": original["identity"], "original_content": original["content_identity"],
        "checkpoint": checkpoint, "policy": policy}))


def _admit(store, identities, first, last, *, full_domain=False, traces=0, retained_bytes=0):
    rows = [store.value_read_rows(identity, start=first, end=last) for identity in identities]
    if any(value is None for value in rows):
        raise ValueError("Streak certification requires explicit original read-bound index maintenance")
    limits = active_resource_limits()
    available = (limits.memory_target_mib - limits.cache_mib) * 1024**2
    # Domain preparation is shared. Subsequent numerical reads hold both typed
    # tables plus bounded sort/join/validation buffers and at most two keys/asset.
    estimate = (max(rows, default=0) * 160 if full_domain else sum(rows) * (192 + 32 * traces)) + retained_bytes
    if estimate > available // 2:
        raise MemoryError("Original streak certification exceeds the admitted memory budget before value reads")
    return estimate


def _tail_coordinates(domain, check_canceled):
    """Select at most two admitted keys/asset with bounded coordinate reductions."""
    dates = domain.times.to_list()
    tail = pl.DataFrame(schema={"time": pl.Date, "asset_id": pl.String})
    complete = set()
    assets = len(domain.asset_ids)
    for begin in reversed(range(0, len(dates), CANONICAL_SESSIONS)):
        check_canceled()
        first, last = dates[begin], dates[min(len(dates), begin + CANONICAL_SESSIONS) - 1]
        selected = domain.grid_lazy().filter(pl.col("time").is_between(first, last))
        if complete:
            selected = selected.filter(~pl.col("asset_id").is_in(sorted(complete)))
        selected = selected.sort("asset_id", "time").group_by("asset_id").tail(2).select("time", "asset_id").collect()
        tail = pl.concat([tail, selected]).sort("asset_id", "time").group_by("asset_id").tail(2).select("time", "asset_id")
        complete = set(tail.group_by("asset_id").len().filter(pl.col("len") == 2)["asset_id"])
        if len(complete) == assets:
            break
    return tail.sort("asset_id", "time")


def _derive_state(store, original, parent, domain, check_canceled, *, domain_bytes):
    keys = _tail_coordinates(domain, check_canceled)
    traces = tuple(original["trace_columns"])
    retained_bytes = (domain_bytes + len(domain.asset_ids) * (512 + 512 * len(traces))
        + sum(len(name.encode()) for name in traces) * 16)
    _admit(store, (), domain.times.min(), domain.times.max(), retained_bytes=retained_bytes)
    selected_parent, selected_output, windows = [], [], []
    dates = domain.times.to_list()
    last_keys = keys.group_by("asset_id").tail(1)
    for begin in range(0, len(dates), CANONICAL_SESSIONS):
        first, last = dates[begin], dates[min(len(dates), begin + CANONICAL_SESSIONS) - 1]
        own = keys.filter(pl.col("time").is_between(first, last))
        if own.is_empty():
            continue
        check_canceled()
        _admit(store, (original["identity"], parent["identity"]), first, last,
            traces=len(traces), retained_bytes=retained_bytes)
        selected_parent.append(own.join(store.read_values(parent["identity"], start=first, end=last,
            include_traces=True), on=["time", "asset_id"], how="left"))
        final = last_keys.filter(pl.col("time").is_between(first, last))
        if not final.is_empty():
            selected_output.append(final.join(store.read_values(original["identity"], start=first, end=last,
                include_traces=True), on=["time", "asset_id"], how="left"))
        windows.append([str(first), str(last)])
    assets = {}
    if selected_parent:
        values = pl.concat(selected_parent).sort("asset_id", "time")
        previous = pl.col("value").shift(1).over("asset_id")
        valid = pl.col("value").is_not_null() & ~pl.col("value").is_nan()
        previous_valid = previous.is_not_null() & ~previous.is_nan()
        values = values.with_columns(pl.when(valid & previous_valid).then(
            (pl.col("value") - previous).sign()).otherwise(None).cast(pl.Int8).alias("direction")
        ).group_by("asset_id").tail(1)
        output = pl.concat(selected_output).rename({"value": "count", **{trace: "run:" + trace for trace in traces}})
        for row in values.join(output, on=["time", "asset_id"], how="left").iter_rows(named=True):
            assets[row["asset_id"]] = {"value": _number(row["value"]), "direction": row["direction"], "count": row["count"],
                "previous_traces": {trace: None if row[trace] is None else str(row[trace]) for trace in traces},
                "run_traces": {trace: None if row["run:" + trace] is None else str(row["run:" + trace]) for trace in traces}}
    state = {"schema": STATE_SCHEMA, "through": original["through"], "traces": list(traces), "assets": assets}
    if not valid_streak_state(state):
        raise ValueError("Original streak tails do not establish a valid checkpoint")
    return state, windows


def plan_maintenance(store, spec, identity, *, original_update, context_id, original_context,
        report_id, report_digest, check_canceled=lambda: None):
    """Certify only a completed resetting streak over its original durable parent.

    This explicit operation consumes bounded original tails and writes admission
    metadata. It never invokes a numerical producer, modifies original receipts,
    certifies holding streaks or guesses missing external-source authority.
    """
    from .proof_maintenance import PLAN_DDL
    check_canceled()
    if (not isinstance(report_id, str) or not report_id or not isinstance(report_digest, str)
            or len(report_digest) != 64 or any(char not in "0123456789abcdef" for char in report_digest)):
        raise ValueError("Streak certification requires an immutable caller report")
    original = store.manifest(identity)
    if original is None:
        raise ValueError("Original streak result is missing")
    key = MaterializationKey.from_dict(original["key"])
    declared = {node.node_id: node for node in spec.nodes}
    node = declared[key.node_id]
    if (node.operator != _OPERATOR or node.parameters.get("reset_on_equal", True) is not True
            or len(parents(node)) != 1 or original["execution_policy"] is not None
            or original.get("proof_rebinding") is not None or original["checkpoint"] is not None
            or original["artifacts"] or original["training_audits"]):
        raise ValueError("Streak certification requires an original resetting generic streak result")
    update = store.describe_update(original_update)
    completed = update["results"].get(context_id, {})
    parent_id = parents(node)[0]
    if (completed.get(node.node_id) != identity or declared[parent_id].node_type != "operator"
            or parent_id not in completed
            or not store.update_covers(original_update, context_id, node_ids=[node.node_id, parent_id])):
        raise ValueError("Streak certification requires original completed update and durable parent authority")
    parent = store.manifest(completed[parent_id])
    if parent is None or parent["trace_columns"] != original["trace_columns"]:
        raise ValueError("Original streak parent channels are missing or changed")
    first, last = date.fromisoformat(original["coverage_start"]), date.fromisoformat(original["through"])
    domain_bytes = _admit(store, (identity, parent["identity"]), first, last, full_domain=True)
    result = store.lookup(key, evidence=False)
    domain = result.panel.domain
    validate_streak_calendar(domain)
    if store.domain_matches(identity, domain) is not True or store.domain_matches(parent["identity"], domain) is not True:
        raise ValueError("Streak certification requires exact original indexed coordinates")
    source = store.lookup(MaterializationKey.from_dict(parent["key"]), domain=domain, evidence=False)
    parameters = original_context.get("parameters", {})
    local = Graph.from_logical_spec(LogicalGraphSpec({node.node_id: node.node_id}, spec.nodes),
        inputs={}, node_bindings={parent_id: source.panel}, parameter_bindings=parameters)
    planner = ExecutionRuntime(node_contexts={node.node_id: hash_mapping({
        "history_policy": None, "information_cutoff": original_context.get("information_cutoff")})})
    if (planner.plan_materialization_keys(local)[node.node_id] != key
            or str(domain.times.min()) != str(original_context["anchor"])):
        raise ValueError("Original streak context and parent bindings do not reproduce its completed key")
    proof = store.interval_identity(parent["identity"], start=first, end=last)
    if proof is None:
        raise ValueError("Streak certification requires original parent interval proof maintenance")
    signature = operator_checkpoint_signature(OPERATOR_REGISTRY.get(node.operator), {"reset_on_equal": True})
    state, windows = _derive_state(store, original, parent, domain, check_canceled, domain_bytes=domain_bytes)
    checkpoint = {"signature": signature, "state": state}
    policy = {"name": "checkpoint_prefix.v2", "parameters": _relevant_parameters(node, parameters),
        "parent_ends": {parent_id: str(last)}, "parent_proofs": {parent_id: proof}}
    replacement = _replacement(original, checkpoint, policy)
    body = {"mode": MODE, "original_identity": identity, "original_content": original["content_identity"],
        "original_key": key.to_dict(), "original_update": original_update, "context_id": context_id,
        "original_context": original_context, "logical_node": node.to_dict(),
        "parent": {"identity": parent["identity"], "content_identity": parent["content_identity"]},
        "checkpoint": checkpoint, "policy": policy, "replacement_key": replacement.to_dict(),
        "read_windows": windows, "report_id": report_id, "report_digest": report_digest}
    body = json.loads(json.dumps(body, sort_keys=True, allow_nan=False))
    plan = {"identity": hash_mapping(body), **body}
    with store.transaction() as connection:
        check_canceled()
        connection.execute(PLAN_DDL)
        encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"))
        connection.execute("INSERT OR IGNORE INTO input_proof_rebinding_plans VALUES(?,?)", (plan["identity"], encoded))
        verify_maintenance(type(store), connection, plan, replacement, policy)
        check_canceled()
    return plan


def verify_maintenance(store_type, connection, reference, key, policy, *, visited=(), memo=None, dependencies=None):
    """Authenticate immutable certification and the complete original ancestry."""
    locator = ("streak_checkpoint_plan", reference["identity"])
    plan = None if memo is None else memo.get(locator, copy_value=False)
    if plan is None:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='input_proof_rebinding_plans'").fetchone()
        row = None if not exists else connection.execute(
            "SELECT plan_json FROM input_proof_rebinding_plans WHERE identity=?", (reference["identity"],)).fetchone()
        if row is None:
            raise ValueError("Streak certification lacks immutable admission")
        plan = json.loads(row[0])
        body = {name: value for name, value in plan.items() if name != "identity"}
        if hash_mapping(body) != plan["identity"] or plan["identity"] != reference["identity"]:
            raise ValueError("Streak certification admission changed")
        if memo is not None:
            memo.put(locator, plan, 2 * len(row[0].encode()))
    handle = {name: plan[name] for name in ("mode", "identity", "original_identity")}
    if (plan["mode"] != MODE or reference not in (plan, handle)
            or plan["policy"] != policy or plan["replacement_key"] != key.to_dict()):
        raise ValueError("Streak certification admission changed")
    original_id = plan["original_identity"]
    if original_id in visited or key.identity == original_id:
        raise ValueError("Streak certification has cyclic ancestry")
    authority = ("streak_checkpoint_authority", plan["identity"])
    checked = None if memo is None else memo.get(authority, copy_value=False)
    if checked is not None:
        if {*visited, key.identity}.intersection(checked):
            raise ValueError("Streak certification has cyclic ancestry")
        if dependencies is not None:
            dependencies.update(checked)
        return plan
    original = store_type._verified_reference_manifest(connection, original_id, plan["original_content"], memo=memo)
    parent = store_type._verified_reference_manifest(connection, plan["parent"]["identity"],
        plan["parent"]["content_identity"], memo=memo)
    node = LogicalNodeSpec.create(**{name: plan["logical_node"][name] for name in
        ("node_type", "input_key", "operator", "parameters", "inputs", "panel_parameters")})
    if (original["key"] != plan["original_key"] or original["execution_policy"] is not None
            or original.get("proof_rebinding") is not None or original["checkpoint"] is not None
            or original["artifacts"] or original["training_audits"]
            or node.to_dict() != plan["logical_node"] or node.node_id != key.node_id
            or node.operator != _OPERATOR or node.parameters.get("reset_on_equal", True) is not True
            or len(parents(node)) != 1 or key != _replacement(original, plan["checkpoint"], policy)
            or original["trace_columns"] != parent["trace_columns"]):
        raise ValueError("Streak certification original numerical authority changed")
    index, parent_index = read_index(connection, original), read_index(connection, parent)
    if index is None or parent_index is None:
        raise ValueError("Streak certification lost original indexed authority")
    checkpoint = plan["checkpoint"]
    if (not valid_streak_state(checkpoint["state"]) or checkpoint["state"]["through"] != original["through"]
            or checkpoint["state"]["traces"] != original["trace_columns"]
            or checkpoint["signature"]["operator"] != _OPERATOR
            or checkpoint["signature"]["parameters"] != {"reset_on_equal": True}):
        raise ValueError("Streak certification checkpoint changed")
    parent_id = parents(node)[0]
    update = store_type._read_update_receipt(connection, plan["original_update"])
    selected = update["results"].get(plan["context_id"], {})
    first, last = date.fromisoformat(original["coverage_start"]), date.fromisoformat(original["through"])
    if (selected.get(node.node_id) != original_id or selected.get(parent_id) != parent["identity"]
            or original["key"]["inputs"] != [parent["identity"]]
            or policy != {"name": "checkpoint_prefix.v2",
                "parameters": _relevant_parameters(node, plan["original_context"].get("parameters", {})),
                "parent_ends": {parent_id: str(last)}, "parent_proofs": {parent_id: interval(parent_index, first, last)}}):
        raise ValueError("Streak certification lost original completed parent proof")
    found = {original_id, parent["identity"]}
    found.update(store_type._verify_reference_sources(connection, parent,
        visiting={*visited, key.identity, original_id}, memo=memo))
    if memo is not None:
        memo.put(authority, tuple(sorted(found)), sum(len(value) for value in found))
    if dependencies is not None:
        dependencies.update(found)
    return plan


def apply_maintenance(store, plan, *, check_canceled=lambda: None):
    """Publish a checkpoint candidate referencing original values, without a kernel."""
    check_canceled()
    key = MaterializationKey.from_dict(plan["replacement_key"])
    with store.read_transaction() as connection:
        verify_maintenance(type(store), connection, plan, key, plan["policy"])
    if store.describe(key.identity) is not None:
        return {"identity": key.identity, "original_identity": plan["original_identity"],
            "outcome": "up_to_date", "computed_sessions": 0}
    original = store.lookup(MaterializationKey.from_dict(plan["original_key"]), evidence=False)
    value = _reference_node(original.panel.lazy(include_traces=True), original.panel.domain, key,
        value_type=original.panel.value_type, trace_columns=original.panel.trace_columns)
    check_canceled()
    store.publish(NodeMaterialization(key, value, checkpoint=plan["checkpoint"]), execution_policy=plan["policy"],
        value_references=(original.key.identity,),
        _proof_rebinding={name: plan[name] for name in ("mode", "identity", "original_identity")})
    return {"identity": key.identity, "original_identity": original.key.identity,
        "outcome": "certified", "computed_sessions": 0}
