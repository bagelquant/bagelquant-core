"""Deterministic pointwise regions with durable boundaries and exact proofs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from .computation_index import describe, interval
from .finite_work import (_run_finite_outputs, admit_finite_payload, dispatch_finite_outputs,
    finite_payload_bytes)
from .graph_management import parents
from .hashing import _hash_dataframe_days, hash_mapping
from .incremental import (CANONICAL_SESSIONS, FINITE_EXECUTION_POLICY, _parent_window,
    _reference_node, _relevant_parameters, scoped_domain)
from .materialization import MaterializationKey, NodeMaterialization, materialization_trace_identity
from .node import Node
from .operation_contract import ExecutionMode
from .operator import OPERATOR_REGISTRY
from .resources import active_resource_limits


@dataclass(frozen=True, slots=True)
class ExecutionRegion:
    """One bounded region; only its output is numerical receipt authority."""
    region_id: str
    output_node: str
    nodes: tuple[str, ...]
    external_inputs: tuple[str, ...]


def plan_regions(spec, *, policies, requested, persist_nodes, causal_nodes):
    declared = {node.node_id: node for node in spec.nodes}
    operators = {key: node for key, node in declared.items() if node.node_type == "operator"}
    if not set(persist_nodes) <= set(operators):
        raise ValueError("persist_nodes must name selected operator nodes")
    eligible = set()
    for key, node in operators.items():
        operation = OPERATOR_REGISTRY.get(node.operator)
        if (key not in causal_nodes and operation.operation.__module__.startswith("bagelquant_core.operator.")
                and operation.contract.execution == ExecutionMode.LAZY and operation.contract.deterministic
                and operation.factory is None and all(key in value and value[key]["history_observations"] == 0
                    for value in policies.values())):
            eligible.add(key)
    children = {key: [] for key in operators}
    for node in operators.values():
        for parent in parents(node):
            if parent in children:
                children[parent].append(node.node_id)
    durable = set(requested) | set(persist_nodes) | (set(operators) - eligible)
    durable.update(key for key, downstream in children.items() if len(downstream) != 1
        or any(child not in eligible for child in downstream))
    regions = []
    for output in operators:
        if output not in durable or output not in eligible:
            continue
        members = {output}
        def include(key):
            for parent in parents(operators[key]):
                if parent in eligible and parent not in durable:
                    members.add(parent)
                    include(parent)
        include(output)
        if len(members) < 2:
            continue
        ordered = tuple(key for key in operators if key in members)
        external = tuple(dict.fromkeys(parent for key in ordered for parent in parents(operators[key]) if parent not in members))
        regions.append(ExecutionRegion(hash_mapping({"nodes": ordered}), output, ordered, external))
    return tuple(regions)


def _block_key(plan, context, node_id, start, end, indexes, direct, *, require_proof=False, metadata_only=False):
    node = plan._nodes[node_id]
    proofs = []
    for parent in parents(node):
        if parent in indexes:
            index = indexes[parent]
            token = interval(index, start, end)
            domain = plan.domains[context][parent]
            kind, traces = index["value_type"], tuple(index["trace_columns"])
            source = plan.keys[context][parent].identity
        else:
            value = direct[parent]
            token = plan._proof_reader(context, metadata_only=metadata_only)(parent, value, start, end)
            if token is None and require_proof:
                if metadata_only:
                    from .finite_work import _MemoryProofUnavailable
                    raise _MemoryProofUnavailable
                raise ValueError("Completed region history requires interval proofs; run explicit owner evidence maintenance")
            domain, kind, traces, source = value.domain, value.value_type.value, value.trace_columns, value.identity
        own = scoped_domain(domain, start, end)
        proofs.append(hash_mapping({"version": FINITE_EXECUTION_POLICY,
            "content": token or {"source": source, "start": str(start), "end": str(end)},
            "domain": own.signature, "type": kind, "traces": traces}))
    policy = plan.policies[context][node_id]
    parameters = _relevant_parameters(node, plan._parameters[context])
    key_context = hash_mapping({"block": policy, "start": str(start), "end": str(end), "parameters": parameters})
    return MaterializationKey(node_id, plan.keys[context][node_id].implementation_id, tuple(proofs),
        scoped_domain(plan.domains[context][node_id], start, end).signature, key_context)


def _matching_certificate(plan, context, region, certificate, start, end, direct, *, require_proof=False, metadata_only=False):
    if certificate.get("boundary_node") != region.output_node or not set(region.nodes) <= set(certificate.get("nodes", {})):
        return False
    indexes = {}
    for node in region.nodes:
        item = certificate["nodes"][node]
        expected = _block_key(plan, context, node, start, end, indexes, direct,
            require_proof=require_proof, metadata_only=metadata_only)
        if expected.to_dict() != item["key"]:
            return False
        index = item["index"]
        if (index["domain_identity"] != expected.domain_identity
                or index["calendar"] != [str(day) for day in scoped_domain(plan.domains[context][node], start, end).times]):
            raise ValueError("region interval evidence differs from canonical coordinates")
        indexes[node] = index
    return True


def execute_region(plan, context, region, guard, *, proof_rebinding=None, _memory_only=False, _direct=None):
    """Execute only missing region intervals with the original block identities."""
    declared = {node.node_id: node for node in plan.spec.nodes}
    direct = _direct if _direct is not None else {
        parent: (plan._inputs[context][declared[parent].input_key] if declared[parent].node_type == "input"
        else plan._values[context][parent]) for parent in region.external_inputs}
    root = region.output_node
    domain = plan.domains[context][root]
    calendar = domain.times.to_list()
    policy = plan.policies[context][root]
    publication_policy = {**policy, "parameter_bindings": _relevant_parameters(plan._nodes[root], plan._parameters[context])}
    completed = plan.store.completed_materializations(plan.graph.graph_id, context, root)
    current = [plan.store.describe(identity) for identity in completed]
    definitions = {node: plan.completion_digests[context][node]["definition_digest"] for node in region.nodes}
    require_proof = False
    for item in current:
        if item["key"]["implementation_id"] != plan.keys[context][root].implementation_id:
            continue
        seals = plan.store.region_evidence(item["identity"], kind="seal")
        if not seals or any(all(seal.get("definitions", {}).get(node) == digest for node, digest in definitions.items()) for seal in seals):
            require_proof = True
    candidates = {}
    for header in plan.store.materialization_candidates(root, plan.keys[context][root].implementation_id,
            policy_key=hash_mapping(policy)):
        for certificate in plan.store.region_evidence(header["identity"], kind="block"):
            if certificate.get("boundary_node") == root and set(region.nodes) <= set(certificate.get("nodes", {})):
                candidates.setdefault(header["coverage_start"], []).append((header, certificate))
    selected = {}
    for begin in range(0, len(calendar), CANONICAL_SESSIONS):
        start, end = calendar[begin], calendar[min(begin + CANONICAL_SESSIONS, len(calendar)) - 1]
        for header, certificate in sorted(candidates.get(str(start), ()), key=lambda item: item[0]["through"], reverse=True):
            prior_end = date.fromisoformat(header["through"])
            if prior_end not in calendar[begin:begin + CANONICAL_SESSIONS] or prior_end > end:
                continue
            if _matching_certificate(plan, context, region, certificate, start, prior_end, direct,
                    require_proof=require_proof, metadata_only=_memory_only):
                selected[begin] = (header, certificate)
                break
    if require_proof and not selected:
        raise ValueError("Input proof changes cover completed region history; resolve impact or run explicit evidence maintenance")
    if proof_rebinding is not None:
        prefix_end = date.fromisoformat(proof_rebinding["repair"]["prefix_through"])
        for begin in range(0, len(calendar), CANONICAL_SESSIONS):
            if calendar[begin] > prefix_end:
                continue
            needed_end = min(prefix_end, calendar[min(begin + CANONICAL_SESSIONS, len(calendar)) - 1])
            retained = selected.get(begin)
            if retained is None or date.fromisoformat(retained[0]["through"]) < needed_end:
                raise ValueError("Cutoff repair requires complete retained region prefix evidence; run explicit evidence maintenance")
    usage = {node: {"computed_blocks": 0, "reused_blocks": 0, "computed_output_sessions": 0,
        "kernel_sessions": 0, "input_sessions": 0, "canonical_sessions": CANONICAL_SESSIONS} for node in region.nodes}
    references, panels = [], []
    peak_payload = 0
    for begin in range(0, len(calendar), CANONICAL_SESSIONS):
        guard()
        start, end = calendar[begin], calendar[min(begin + CANONICAL_SESSIONS, len(calendar)) - 1]
        retained = selected.get(begin)
        prefix = None
        if retained is not None:
            header, certificate = retained
            old_end = date.fromisoformat(header["through"])
            key = MaterializationKey.from_dict(certificate["nodes"][root]["key"])
            if not _memory_only:
                prefix = plan.store.lookup(key, domain=scoped_domain(domain, start, old_end), evidence=False)
            if old_end == end:
                if _memory_only:
                    continue
                references.append(prefix.key.identity)
                panels.append(prefix.panel)
                for node in region.nodes:
                    usage[node]["reused_blocks"] += 1
                continue
            output_start = calendar[calendar.index(old_end) + 1]
        else:
            output_start = start
        if proof_rebinding is not None and str(output_start) < proof_rebinding["repair"]["from"]:
            raise ValueError("Cutoff repair cannot rerun retained historical region outputs")
        traces = tuple(dict.fromkeys(trace for value in direct.values() for trace in value.trace_columns))
        estimate = finite_payload_bytes([
            *((scoped_domain(value.domain, output_start, end), value.trace_columns) for value in direct.values()),
            *((scoped_domain(plan.domains[context][node], output_start, end), traces) for node in region.nodes)])
        if _memory_only:
            peak_payload = max(peak_payload, estimate)
            continue
        admit_finite_payload(estimate)
        if plan._input_window_loader is None:
            for member in region.nodes:
                plan._load_inputs(context, member, guard, active_resource_limits())
            direct = {parent: (plan._inputs[context][declared[parent].input_key] if declared[parent].node_type == "input"
                else plan._values[context][parent]) for parent in region.external_inputs}
        scoped = {}
        for parent, value in direct.items():
            frames = _parent_window(parent, value, output_start, end,
                input_window_loader=plan._window_reader(context, guard, active_resource_limits()),
                stored_parent=declared[parent].node_type == "operator", check_canceled=guard)
            own = scoped_domain(value.domain, output_start, end)
            scoped[parent] = Node.from_domain(frames, own, identity=value.identity, value_type=value.value_type,
                trace_columns=value.trace_columns, trace_identity=value.trace_identity)
        dispatch = plan._kernel_dispatch
        predicate = getattr(dispatch, "should_dispatch", None)
        if dispatch is not None and (predicate is None or predicate(estimate)):
            outputs = dispatch_finite_outputs(plan.spec, root, region.nodes, scoped,
                plan._parameters[context], output_start, dispatch, guard, estimated_payload_bytes=estimate)
        else:
            outputs = _run_finite_outputs(plan.spec, region.nodes, scoped,
                plan._parameters[context], output_start, guard)
        indexes, evidence, root_panel = {}, {}, None
        for node in region.nodes:
            frame, kind, traces = outputs[node]
            missing_domain = scoped_domain(plan.domains[context][node], output_start, end)
            typed = Node.from_domain(frame, missing_domain, identity=plan.keys[context][node].identity,
                value_type=kind, trace_columns=traces)
            frame = typed.collect(dense=False, include_traces=True)
            days = _hash_dataframe_days(frame)
            tail = describe(missing_domain, days=days, value_type=typed.value_type.value, trace_columns=traces)
            if prefix is not None:
                original = certificate["nodes"][node]["index"]
                index = {**tail, "calendar": original["calendar"] + tail["calendar"],
                    "membership": {**original["membership"], **tail["membership"]},
                    "days": {**original["days"], **tail["days"]},
                    "domain_identity": scoped_domain(plan.domains[context][node], start, end).signature}
            else:
                index = tail
            key = _block_key(plan, context, node, start, end, indexes, direct, require_proof=require_proof)
            indexes[node] = index
            evidence[node] = {"key": key.to_dict(), "index": index}
            usage[node]["computed_blocks"] += 1
            usage[node]["computed_output_sessions"] += len(missing_domain.times)
            usage[node]["kernel_sessions"] += len(missing_domain.times)
            usage[node]["input_sessions"] += len(missing_domain.times) * len(parents(plan._nodes[node]))
            if node == root:
                root_panel = typed
        guard()
        block_domain = scoped_domain(domain, start, end)
        block_key = MaterializationKey.from_dict(evidence[root]["key"])
        proof = {"region_id": region.region_id, "boundary_node": root, "nodes": evidence}
        existing = plan.store.describe(block_key.identity)
        if existing is not None:
            # Changed transient inputs can yield the exact same durable key.
            # Retain that complete result rather than compose a cycle through
            # a shorter prefix that already references it. Add only the new
            # content-bound logical certificate; original evidence is immutable.
            from .computation_index import read
            with plan.store.transaction() as connection:
                guard()
                row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (block_key.identity,)).fetchone()
                import json
                manifest = json.loads(row[0])
                plan.store._verify_metadata(manifest, block_key.identity)
                plan.store._verify_artifact_sources(connection, manifest)
                if read(connection, manifest) != evidence[root]["index"]:
                    raise ValueError("Retained region block differs from its recomputed bounded evidence")
                plan.store._save_region_evidence(connection, manifest, {"block": proof})
        elif prefix is None:
            value = Node.from_domain(root_panel.lazy(include_traces=True), block_domain, identity=block_key.identity,
                value_type=root_panel.value_type, trace_columns=root_panel.trace_columns,
                trace_identity=materialization_trace_identity(block_key, root_panel.trace_columns))
            plan.store.publish(NodeMaterialization(block_key, value), execution_policy=publication_policy,
                region_evidence={"block": proof})
        else:
            fragment_key = MaterializationKey(root, block_key.implementation_id, block_key.inputs,
                root_panel.domain.signature, hash_mapping({"fragment_of": block_key.identity, "start": str(output_start)}))
            fragment = Node.from_domain(root_panel.lazy(include_traces=True), root_panel.domain, identity=fragment_key.identity,
                value_type=root_panel.value_type, trace_columns=root_panel.trace_columns,
                trace_identity=materialization_trace_identity(fragment_key, root_panel.trace_columns))
            plan.store.publish(NodeMaterialization(fragment_key, fragment), execution_policy=publication_policy)
            value = _reference_node(pl.concat([prefix.panel.lazy(include_traces=True), fragment.lazy(include_traces=True)]),
                block_domain, block_key, value_type=fragment.value_type, trace_columns=fragment.trace_columns)
            plan.store.publish(NodeMaterialization(block_key, value), execution_policy=publication_policy,
                value_references=(prefix.key.identity, fragment_key.identity), region_evidence={"block": proof})
        record = plan.store.lookup(block_key, domain=block_domain, evidence=False)
        references.append(record.key.identity)
        panels.append(record.panel)
    if _memory_only:
        return peak_payload * 8
    key = plan.keys[context][root]
    result = _reference_node(pl.concat([panel.lazy(include_traces=True) for panel in panels]), domain,
        key, value_type=panels[0].value_type, trace_columns=panels[0].trace_columns)
    seal = {"region_id": region.region_id, "boundary_node": root,
        "nodes": {node: plan.keys[context][node].to_dict() for node in region.nodes},
        "definitions": definitions, "blocks": references}
    guard()
    plan.store.publish(NodeMaterialization(key, result), execution_policy=publication_policy,
        value_references=references, region_evidence={"seal": seal}, _proof_rebinding=proof_rebinding)
    return plan.store.lookup(key, domain=domain, evidence=False), usage
