"""Explicit causal cutoff repair using ordinary bounded Runtime work/publication.

The admission preserves existing numerical policies. It is deliberately limited
to a changed final session, exact built-in backward support and a proven prefix.
It does not turn unknown generic history into an ordinary append contract.
"""
from __future__ import annotations

from datetime import date
import json

import polars as pl

from .hashing import hash_mapping
from .graph_management import parents
from .materialization import MaterializationKey, NodeMaterialization
from .node import Node
from .incremental import _numerical_read_start, _parent_window, _reference_node


def repair_history(node):
    """Exact support for this maintenance admission, without changing old keys."""
    from .operator import OPERATOR_REGISTRY
    from .operation_contract import TraceRule
    operation = OPERATOR_REGISTRY.get(node.operator)
    function = operation.operation
    module, name = function.__module__, function.__name__
    if (not module.startswith("bagelquant_core.operator.") or operation.factory is not None
            or not operation.contract.deterministic):
        raise ValueError("Cutoff repair requires a supported causal built-in")
    if (operation.history(node.parameters) == 0 and operation.contract.trace_rule in
            {TraceRule.NONE, TraceRule.PASSTHROUGH, TraceRule.PARENT_MAX}):
        return 0
    if module == "bagelquant_core.operator.streaks" and name in {"diff_from_last_change", "pct_change_from_last_change"}:
        return 1
    if module == "bagelquant_core.operator.missing" and name == "ffill":
        limit = node.parameters.get("limit")
        if isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0:
            return limit
    raise ValueError("Cutoff repair requires exact bounded backward support; this operator remains blocked")


def admit_repair(store, proof, plan, guard):
    from .cutoff_proof import verify_normalization
    guard()
    if plan is None or plan.store is not store:
        raise ValueError("Cutoff repair must be admitted into its ordinary frozen UpdatePlan")
    if plan._input_loader is not None and plan._input_window_loader is None:
        raise ValueError("Cutoff repair requires bounded source loading; full-history loaders remain blocked")
    plan.guard()
    context = proof["context_id"]
    if (proof.get("region_prefix") is not None or plan.spec.to_dict()["nodes"] != proof["spec"]["nodes"]
            or plan.state["contexts"].get(context) != proof["replacement_context"]
            or plan.through != proof["through"]
            or any(plan.keys[context][node].to_dict() != item["replacement_key"] for node, item in proof["nodes"].items())):
        raise ValueError("Cutoff repair differs from the frozen update authority")
    key_node = next(node for node, item in proof["nodes"].items() if item.get("original_identity") == proof["original_identity"])
    with store.read_transaction() as connection:
        verify_normalization(type(store), connection, proof, plan.keys[context][key_node], proof["nodes"][key_node]["policy"])
    if set(plan._durable_nodes) != {node for node, item in proof["nodes"].items() if "original_identity" in item}:
        raise ValueError("Cutoff repair must preserve the original durable boundaries")
    guard()
    plan.guard()
    plan._input_proof_repairs[context] = proof
    return {"outcome": "admitted", "computed_sessions": 0, "repair_from": proof["repair"]["from"]}


def execute_repair(plan, context, node, guard):
    """Called only inside UpdatePlan's ordinary admission, limits and fences."""
    from .cutoff_proof import verify_normalization
    from .finite_work import _run_finite_outputs, dispatch_finite_outputs, finite_payload_bytes, admit_finite_payload
    from .resources import active_resource_limits
    proof = plan._input_proof_repairs[context]
    key, item = plan.keys[context][node], proof["nodes"][node]
    with plan.store.read_transaction() as connection:
        verify_normalization(type(plan.store), connection, proof, key, item["policy"])
    if node in plan._region_outputs:
        from .regions import execute_region
        region = plan._region_outputs[node]
        _prepare_region_prefix(plan, context, region, proof, guard)
        return execute_region(plan, context, region, guard, proof_rebinding=proof)
    start, end = date.fromisoformat(proof["repair"]["from"]), date.fromisoformat(plan.through)
    prefix_end = date.fromisoformat(proof["repair"]["prefix_through"])
    domain = plan.domains[context][node]
    tail_domain = domain.window(start, end)
    fragment_key = MaterializationKey(node, key.implementation_id, key.inputs, tail_domain.signature,
        hash_mapping({"cutoff_repair": proof["identity"], "result": key.identity, "from": str(start)}))
    usage = {"computed_blocks": 0, "reused_blocks": 1, "computed_output_sessions": 0,
        "kernel_sessions": 0, "input_sessions": 0}
    if plan.store.describe(fragment_key.identity) is None:
        declared = {value.node_id: value for value in plan.spec.nodes}
        direct = {parent: (plan._inputs[context][declared[parent].input_key] if declared[parent].node_type == "input"
            else plan._values[context][parent]) for parent in parents(plan._nodes[node])}
        history = proof["repair"]["history"][node]
        # Membership halo queries allocate coordinate scratch, before any values.
        admit_finite_payload(finite_payload_bytes([(value.domain, ()) for value in direct.values()
            if history and value.domain.is_dynamic]))
        read_start = min(_numerical_read_start(value.domain, start, end, history) for value in direct.values())
        windows = {parent: value.domain.window(read_start, end) for parent, value in direct.items()}
        estimate = finite_payload_bytes([*( (windows[parent], value.trace_columns) for parent, value in direct.items()),
            (domain.window(read_start, end), tuple(dict.fromkeys(trace for value in direct.values() for trace in value.trace_columns)))])
        admit_finite_payload(estimate)
        scoped = {}
        reader = plan._window_reader(context, guard, active_resource_limits(), bounded_stored_parents=True)
        for parent, value in direct.items():
            frame = _parent_window(parent, value, read_start, end, input_window_loader=reader,
                stored_parent=declared[parent].node_type == "operator", check_canceled=guard)
            scoped[parent] = Node.from_domain(frame, windows[parent], identity=value.identity,
                value_type=value.value_type, trace_columns=value.trace_columns, trace_identity=value.trace_identity)
        dispatch = plan._kernel_dispatch
        predicate = getattr(dispatch, "should_dispatch", None)
        if dispatch is not None and (predicate is None or predicate(estimate)):
            outputs = dispatch_finite_outputs(plan.spec, node, (node,), scoped, plan._parameters[context],
                start, dispatch, guard, estimated_payload_bytes=estimate)
        else:
            outputs = _run_finite_outputs(plan.spec, (node,), scoped, plan._parameters[context], start, guard)
        frame, kind, traces = outputs[node]
        value = _reference_node(frame.lazy(), tail_domain, fragment_key, value_type=kind, trace_columns=traces)
        guard()
        plan.store.publish(NodeMaterialization(fragment_key, value), execution_policy=item["policy"])
        usage.update(computed_blocks=1, computed_output_sessions=len(tail_domain.times),
            kernel_sessions=len(domain.window(read_start, end).times),
            input_sessions=sum(len(own.times) for own in windows.values()))
    original = plan.store.lookup(MaterializationKey.from_dict(item["original_key"]), evidence=False)
    fragment = plan.store.lookup(fragment_key, domain=tail_domain, evidence=False)
    value = _reference_node(pl.concat([original.panel.lazy(include_traces=True).filter(pl.col("time") <= prefix_end),
        fragment.panel.lazy(include_traces=True)]), domain, key,
        value_type=fragment.panel.value_type, trace_columns=fragment.panel.trace_columns)
    guard()
    plan.store.publish(NodeMaterialization(key, value), execution_policy=item["policy"], _proof_rebinding=proof,
        value_references=({"identity": original.key.identity, "start": domain.times.min(), "end": prefix_end}, fragment_key.identity))
    return plan.store.lookup(key, domain=domain, evidence=False), {node: usage}


def _clipped(index, domain):
    days = {str(day) for day in domain.times}
    return {**index, "calendar": [str(day) for day in domain.times], "domain_identity": domain.signature,
        "membership": {day: token for day, token in index["membership"].items() if day in days},
        "days": {day: token for day, token in index["days"].items() if day in days}}


def _prepare_region_prefix(plan, context, region, proof, guard):
    """Clip only the final original block; ordinary regions reuse all other blocks."""
    from .regions import _block_key
    from .cutoff_proof import verify_normalization
    from .proof_maintenance import PLAN_DDL
    root = region.output_node
    prefix_end = date.fromisoformat(proof["repair"]["prefix_through"])
    original = proof["nodes"][root]
    declared = {node.node_id: node for node in plan.spec.nodes}
    direct = {parent: (plan._inputs[context][declared[parent].input_key] if declared[parent].node_type == "input"
        else plan._values[context][parent]) for parent in region.external_inputs}
    for identity in original["original_seal"]["blocks"]:
        header = plan.store.describe(identity)
        if header is None:
            raise ValueError("Cutoff repair requires complete retained region prefix evidence; run explicit evidence maintenance")
        first, last = date.fromisoformat(header["coverage_start"]), date.fromisoformat(header["through"])
        if not first <= prefix_end < last:
            continue
        guard()
        certificate = next(value for value in plan.store.region_evidence(identity, kind="block")
            if set(region.nodes) <= set(value["nodes"]))
        indexes, nodes = {}, {}
        for member in region.nodes:
            own = plan.domains[context][member].window(first, prefix_end)
            index = _clipped(certificate["nodes"][member]["index"], own)
            key = _block_key(plan, context, member, first, prefix_end, indexes, direct, require_proof=True)
            indexes[member] = index
            nodes[member] = {"key": key.to_dict(), "index": index}
        replacement = {**certificate, "nodes": nodes}
        key = MaterializationKey.from_dict(nodes[root]["key"])
        if plan.store.describe(key.identity) is not None:
            continue
        body = {name: value for name, value in proof.items() if name != "identity"}
        body["region_prefix"] = {"node": root, "original": identity, "original_content": header["content_identity"],
            "certificate": certificate, "replacement": replacement, "start": str(first), "end": str(prefix_end)}
        admission = {"identity": hash_mapping(body), **body}
        with plan.store.transaction() as connection:
            guard()
            verify_normalization(type(plan.store), connection, admission, key, original["policy"], require_admission=False)
            connection.execute(PLAN_DDL)
            connection.execute("INSERT OR IGNORE INTO input_proof_rebinding_plans VALUES(?,?)",
                (admission["identity"], json.dumps(admission, sort_keys=True, separators=(",", ":"), allow_nan=False)))
        source = plan.store.lookup(MaterializationKey.from_dict(certificate["nodes"][root]["key"]), evidence=False)
        own = plan.domains[context][root].window(first, prefix_end)
        value = _reference_node(source.panel.lazy(include_traces=True).filter(pl.col("time") <= prefix_end), own, key,
            value_type=source.panel.value_type, trace_columns=source.panel.trace_columns)
        guard()
        plan.store.publish(NodeMaterialization(key, value), execution_policy=original["policy"],
            value_references=({"identity": identity, "start": first, "end": prefix_end},),
            region_evidence={"block": replacement}, _proof_rebinding=admission)


def verify_region_prefix(store_type, connection, plan, key, policy):
    """Verify the immutable clipped certificate against its actual original block."""
    from .computation_index import read
    prefix = plan["region_prefix"]
    item = plan["nodes"][prefix["node"]]
    if (plan.get("repair") is None or key.node_id != prefix["node"] or policy != item["policy"]
            or prefix["original"] not in item["original_seal"]["blocks"]
            or prefix["end"] != plan["repair"]["prefix_through"]
            or prefix["start"] > prefix["end"]):
        raise ValueError("Cutoff region prefix lacks original closure authority")
    original = store_type._verified_reference_manifest(connection, prefix["original"], prefix["original_content"])
    rows = store_type._region_evidence_rows(connection, original, "block")
    if (original["content_identity"] != prefix["original_content"]
            or prefix["certificate"] not in store_type._decode_region_evidence(original, "block", rows)
            or prefix["certificate"]["nodes"][key.node_id]["index"] != read(connection, original)
            or prefix["start"] != original["coverage_start"] or prefix["end"] >= original["through"]):
        raise ValueError("Cutoff region prefix original evidence changed")
    before, after = prefix["certificate"], prefix["replacement"]
    if (set(before["nodes"]) != set(after["nodes"]) or after["nodes"][key.node_id]["key"] != key.to_dict()
            or {k: v for k, v in before.items() if k != "nodes"} != {k: v for k, v in after.items() if k != "nodes"}):
        raise ValueError("Cutoff region prefix changed its logical coverage")
    for node, value in after["nodes"].items():
        prior = before["nodes"][node]
        index = prior["index"]
        days = [day for day in index["calendar"] if prefix["start"] <= day <= prefix["end"]]
        expected = {**index, "calendar": days, "domain_identity": value["key"]["domain_identity"],
            "membership": {day: index["membership"][day] for day in days},
            "days": {day: token for day, token in index["days"].items() if day in days}}
        if (value["index"] != expected or not days or days[0] != prefix["start"] or days[-1] != prefix["end"]
                or any(prior["key"][field] != value["key"][field] for field in ("node_id", "implementation_id"))):
            raise ValueError("Cutoff region prefix changed original coordinates or values")
