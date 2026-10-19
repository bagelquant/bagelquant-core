"""Source-bound cutoff normalization and explicit final-session repair admission.

Exact normalization references a completely equivalent interval. A distinct
repair admission proves an unchanged prefix and delegates its final-session
work to the ordinary UpdatePlan. Neither changes numerical execution policies.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json

from .hashing import hash_mapping
from .logical import LogicalGraphSpec
from .materialization import MaterializationKey, NodeMaterialization
from .graph_management import parents


MODE = "cutoff_normalization.v1"
REPAIR_MODE = "cutoff_tail_repair.v1"


def _descriptor(value):
    if getattr(value, "_materialization_identity", None) is not None:
        raise ValueError("Cutoff normalization requires original external source bindings")
    return {"input": value.identity, "trace": value.trace_identity,
        "trace_columns": list(value.trace_columns), "domain": value.domain.signature,
        "categorical": value.value_type == "category", "prediction": value.value_type == "prediction"}


def _projection(value):
    return {**asdict(value), "start": str(value.start), "end": str(value.end)}


def plan_normalization(store, spec, identity, *, original_update, context_id,
        original_context, replacement_context, original_inputs, replacement_inputs,
        original_projection, replacement_projection, report_id, report_digest,
        check_canceled=lambda: None, repair_from=None):
    """Admit exact reuse or a bounded causal tail from source-owned projections."""
    from .proof_maintenance import ConsumedInputEvidence, PLAN_DDL
    from .update import _plan_context_materializations, _same_coordinates
    from .operator import OPERATOR_REGISTRY
    check_canceled()
    update = store.describe_update(original_update)
    operators = {node.node_id: node for node in spec.nodes if node.node_type == "operator"}
    if not store.update_covers(original_update, context_id, node_ids=list(operators)):
        raise ValueError("Original cutoff normalization update does not cover the complete closure")
    results = {node: value for node, value in update["results"][context_id].items() if node in operators}
    coverage = {node: boundary for node, boundary in update["coverage"].get(context_id, {}).items() if node in operators}
    if identity not in results.values() or any(boundary not in results for boundary in coverage.values()):
        raise ValueError("Cutoff normalization requires the original durable closure")
    for node in operators.values():
        operation = OPERATOR_REGISTRY.get(node.operator)
        if not operation.contract.deterministic or operation.factory is not None:
            raise ValueError("Cutoff normalization requires deterministic stateless operators")
    plans = [_plan_context_materializations(spec, inputs, context, capture_contexts=True)
        for inputs, context in ((original_inputs, original_context), (replacement_inputs, replacement_context))]
    if plans[0][1] != plans[1][1]:
        raise ValueError("Cutoff normalization changed numerical policy")
    repair = None
    if repair_from is not None:
        from .cutoff_repair import repair_history
        calendars = [value.domain.times for value in original_inputs.values()]
        calendars += list(plans[0][3].node_domains[node].times for node in operators)
        calendar = calendars[0]
        if (str(repair_from) != update["through"] or len(calendar) < 2
                or str(calendar.max()) != str(repair_from)
                or any(not value.equals(calendar) for value in calendars)):
            raise ValueError("Cutoff repair requires one final session and a shared nonempty retained prefix")
        repair = {"from": str(repair_from), "prefix_through": str(calendar[-2]),
            "history": {node: repair_history(value) for node, value in operators.items()}}
    sources = {}
    input_keys = {node.input_key for node in spec.nodes if node.node_type == "input"}
    if set(original_inputs) != input_keys or set(replacement_inputs) != input_keys:
        raise ValueError("Cutoff normalization requires every external input")
    for name in sorted(input_keys):
        check_canceled()
        before, after = original_inputs[name], replacement_inputs[name]
        if (not _same_coordinates(before.domain, after.domain) or before.value_type != after.value_type
                or before.trace_columns != after.trace_columns):
            raise ValueError("Cutoff normalization changed consumed coordinates, type or traces")
        first = before.domain.times.min()
        last = before.domain.times.max() if repair is None else date.fromisoformat(repair["prefix_through"])
        projection_domain = before.domain if repair is None else before.domain.window(first, last)
        evidence = []
        for callback, value in ((original_projection, before), (replacement_projection, after)):
            proof = callback(name, value, first, last)
            if (not isinstance(proof, ConsumedInputEvidence) or proof.start != first or proof.end != last
                    or proof.domain_signature != projection_domain.signature
                    or proof.value_type != value.value_type.value or proof.trace_columns != value.trace_columns):
                raise ValueError("Cutoff normalization requires complete source-owned consumed evidence")
            evidence.append(_projection(proof))
        sources[name] = {"original": _descriptor(before), "replacement": _descriptor(after),
            "original_projection": evidence[0], "replacement_projection": evidence[1]}
        if repair is not None:
            sources[name]["projection_domain"] = projection_domain.signature
    nodes = {}
    for node_id in operators:
        check_canceled()
        old_key, new_key = plans[0][4][node_id], plans[1][4][node_id]
        if not _same_coordinates(plans[0][3].node_domains[node_id], plans[1][3].node_domains[node_id]):
            raise ValueError("Cutoff normalization changed an operator Domain")
        nodes[node_id] = {"original_key": old_key.to_dict(), "replacement_key": new_key.to_dict(),
            "original_context": plans[0][3].materialization_contexts[node_id],
            "replacement_context": plans[1][3].materialization_contexts[node_id]}
        if node_id in results:
            manifest = store.manifest(results[node_id])
            if manifest is None or manifest["key"] != old_key.to_dict():
                raise ValueError("Original source/context bindings do not reproduce the completed key")
            if store.domain_matches(results[node_id], plans[1][3].node_domains[node_id]) is not True:
                raise ValueError("Cutoff normalization requires exact original indexed coordinates")
            if manifest["artifacts"] or manifest["checkpoint"] is not None or manifest["training_audits"]:
                raise ValueError("Cutoff normalization cannot replace stateful or auxiliary result evidence")
            seals = store.region_evidence(results[node_id], kind="seal")
            required = {node for node, boundary in coverage.items() if boundary == node_id}
            seal = next((seal for seal in seals if required | {node_id} <= set(seal["nodes"])
                and all(seal["nodes"][node] == plans[0][4][node].to_dict() for node in seal["nodes"] if node in operators)), None)
            if required and seal is None:
                raise ValueError("Cutoff normalization requires original logical-region evidence")
            if seal is not None and not set(seal["nodes"]) <= set(operators):
                raise ValueError("Cutoff normalization must include the entire original region")
            nodes[node_id].update(original_identity=results[node_id], original_content=manifest["content_identity"],
                policy=manifest["execution_policy"], original_seal=seal,
                replacement_seal=None if seal is None else {**seal,
                    "nodes": {node: plans[1][4][node].to_dict() for node in seal["nodes"]}})
            if seal is not None:
                from .computation_index import describe
                with store.read_transaction() as connection:
                    coordinates = _verify_region_blocks(type(store), connection, manifest, seal)
                for member in seal["nodes"]:
                    expected = describe(plans[1][3].node_domains[member], days={}, value_type="", trace_columns=())
                    if coordinates[member] != hash_mapping({field: expected[field] for field in ("calendar", "membership")}):
                        raise ValueError("Cutoff normalization changed original region member coordinates")
                nodes[node_id]["region_coordinates"] = coordinates
    body = {"mode": MODE if repair is None else REPAIR_MODE, "original_identity": identity, "original_update": original_update,
        "context_id": context_id, "through": update["through"], "spec": spec.to_dict(),
        "original_context": original_context, "replacement_context": replacement_context,
        "policies": plans[0][1], "sources": sources, "nodes": nodes, "coverage": coverage,
        "report_id": report_id, "report_digest": report_digest}
    if repair is not None:
        body["repair"] = repair
    # Normalize tuples and dates to the same JSON representation as admission.
    body = json.loads(json.dumps(body, sort_keys=True, allow_nan=False))
    plan = {"identity": hash_mapping(body), **body}
    with store.transaction() as connection:
        check_canceled()
        key = MaterializationKey.from_dict(next(item["replacement_key"] for item in nodes.values()
            if item.get("original_identity") == identity))
        policy = next(item["policy"] for item in nodes.values() if item.get("original_identity") == identity)
        verify_normalization(type(store), connection, plan, key, policy, require_admission=False)
        connection.execute(PLAN_DDL)
        connection.execute("INSERT OR IGNORE INTO input_proof_rebinding_plans VALUES(?,?)",
            (plan["identity"], json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)))
        check_canceled()
    return plan


def verify_normalization(store_type, connection, plan, key, policy, *, visited=(), require_admission=True):
    """Recheck original authority, canonical recipes and admitted source equality."""
    from .proof_maintenance import ConsumedInputEvidence
    body = {name: value for name, value in plan.items() if name != "identity"}
    if plan.get("identity") != hash_mapping(body):
        raise ValueError("Cutoff normalization admission checksum differs")
    if require_admission:
        row = connection.execute("SELECT plan_json FROM input_proof_rebinding_plans WHERE identity=?", (plan["identity"],)).fetchone()
        if row is None or json.loads(row[0]) != plan:
            raise ValueError("Cutoff normalization lacks immutable source admission")
    original_update = store_type._read_update_receipt(connection, plan["original_update"])
    original_results = original_update["results"].get(plan["context_id"], {})
    original_coverage = original_update["coverage"].get(plan["context_id"], {})
    if (original_update["through"] != plan["through"]
            or any(original_results.get(node) != item["original_identity"]
                for node, item in plan["nodes"].items() if "original_identity" in item)
            or any(original_coverage.get(node) != boundary for node, boundary in plan["coverage"].items())):
        raise ValueError("Cutoff normalization lost original update authority")
    if (not plan["report_id"] or len(plan["report_digest"]) != 64
            or any(char not in "0123456789abcdef" for char in plan["report_digest"])):
        raise ValueError("Cutoff normalization requires an immutable caller report")
    old, new = plan["original_context"], plan["replacement_context"]
    if ({name: value for name, value in old.items() if name != "information_cutoff"}
            != {name: value for name, value in new.items() if name != "information_cutoff"}
            or not plan["through"] <= new["information_cutoff"] < old["information_cutoff"]):
        raise ValueError("Cutoff normalization only permits a lower cutoff with the same complete output interval")
    spec = LogicalGraphSpec.from_dict(plan["spec"])
    repair = plan.get("repair")
    if plan["mode"] == REPAIR_MODE:
        from .cutoff_repair import repair_history
        if (not repair or repair["from"] != plan["through"] or repair["prefix_through"] >= repair["from"]
                or repair["history"] != {node.node_id: repair_history(node) for node in spec.nodes if node.node_type == "operator"}):
            raise ValueError("Cutoff repair lost its bounded causal support contract")
    elif repair is not None:
        raise ValueError("Cutoff normalization cannot carry an undeclared repair")
    physical = {"original": {}, "replacement": {}}
    for node in spec.nodes:
        if node.node_type == "input":
            source = plan["sources"][node.input_key]
            proofs = []
            for label in physical:
                raw = source[label + "_projection"]
                proof = ConsumedInputEvidence(**{**raw, "start": date.fromisoformat(raw["start"]),
                    "end": date.fromisoformat(raw["end"]), "trace_columns": tuple(raw["trace_columns"])})
                descriptor = source[label]
                if (proof.end.isoformat() != (plan["through"] if repair is None else repair["prefix_through"])
                        or proof.domain_signature != (descriptor["domain"] if repair is None else source["projection_domain"])
                        or list(proof.trace_columns) != descriptor["trace_columns"]
                        or (proof.value_type == "category") != descriptor["categorical"]
                        or (proof.value_type == "prediction") != descriptor["prediction"]):
                    raise ValueError("Cutoff normalization source scope/type/trace contract differs")
                physical[label][node.node_id] = hash_mapping(descriptor)
                proofs.append(raw)
            if any(proofs[0][field] != proofs[1][field] for field in
                    ("content_token", "start", "end", "domain_signature", "value_type", "trace_columns")):
                raise ValueError("Consumed historical inputs changed or lack exact equivalence")
            continue
        item = plan["nodes"][node.node_id]
        before, after = item["original_key"], item["replacement_key"]
        if any(before[name] != after[name] for name in ("node_id", "implementation_id", "domain_identity")):
            raise ValueError("Cutoff normalization changed the numerical contract")
        recipes = [item[label + "_context"] for label in physical]
        if any(recipes[0][field] != recipes[1][field] for field in ("operator", "parameters")):
            raise ValueError("Cutoff normalization changed operator state, calendar or parameters")
        for label, context in (("original", old), ("replacement", new)):
            recipe, candidate = item[label + "_context"], item[label + "_key"]
            if (recipe["context"] != hash_mapping({"history_policy": plan["policies"].get(node.node_id),
                    "information_cutoff": context["information_cutoff"]})
                    or candidate["context_identity"] != hash_mapping(recipe)
                    or candidate["inputs"] != [physical[label][parent] for parent in parents(node)]):
                raise ValueError("Cutoff normalization does not reproduce the canonical complete key")
            physical[label][node.node_id] = MaterializationKey.from_dict(candidate).identity
        original_id = item.get("original_identity")
        if original_id is None:
            if node.node_id not in plan["coverage"]:
                raise ValueError("Cutoff normalization omitted original node authority")
            continue
        if original_id in visited or original_id == key.identity:
            raise ValueError("Cutoff normalization contains a reference cycle")
        row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (original_id,)).fetchone()
        if row is None:
            raise ValueError("Cutoff normalization original materialization is missing")
        manifest = json.loads(row[0])
        store_type._verify_metadata(manifest, original_id)
        if (manifest["key"] != before or manifest["content_identity"] != item["original_content"]
                or manifest["execution_policy"] != item["policy"] or manifest["through"] != plan["through"]
                or manifest["artifacts"] or manifest["checkpoint"] is not None or manifest["training_audits"]):
            raise ValueError("Cutoff normalization original authority changed")
        if manifest.get("proof_rebinding") is not None:
            from .proof_maintenance import verify_rebinding
            verify_rebinding(store_type, connection, manifest["proof_rebinding"],
                MaterializationKey.from_dict(before), manifest["execution_policy"], visited=(*visited, original_id))
        seal = item["original_seal"]
        if seal is not None:
            rows = connection.execute("SELECT content_identity,payload_json,payload_identity FROM region_evidence "
                "WHERE identity=? AND kind='seal'", (original_id,)).fetchall()
            if seal not in store_type._decode_region_evidence(manifest, "seal", rows) or any(
                    seal["nodes"][member] != plan["nodes"][member]["original_key"] for member in seal["nodes"]):
                raise ValueError("Cutoff normalization lost original region authority")
            coordinates = _verify_region_blocks(store_type, connection, manifest, seal,
                visited=(*visited, original_id))
            if coordinates != item["region_coordinates"]:
                raise ValueError("Cutoff normalization region coordinates changed")
            replacement = {**seal, "nodes": {member: plan["nodes"][member]["replacement_key"] for member in seal["nodes"]}}
            if item["replacement_seal"] != replacement:
                raise ValueError("Cutoff normalization changed retained region evidence")
    if plan.get("region_prefix") is not None:
        from .cutoff_repair import verify_region_prefix
        return verify_region_prefix(store_type, connection, plan, key, policy)
    selected = plan["nodes"].get(key.node_id)
    if (selected is None or selected.get("original_identity") is None
            or selected["replacement_key"] != key.to_dict() or selected["policy"] != policy):
        raise ValueError("Cutoff normalization does not admit this replacement")


def _verify_region_blocks(store_type, connection, manifest, seal, *, visited=()):
    """Retain content-bound block certificates and their full original coverage."""
    from .computation_index import read
    full = read(connection, manifest)
    if full is None:
        raise ValueError("Cutoff normalization needs original region coordinate evidence")
    calendar = []
    coordinates = {node: {"calendar": [], "membership": {}} for node in seal["nodes"]}
    for identity in seal["blocks"]:
        row = connection.execute("SELECT manifest_json FROM materializations WHERE identity=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("Cutoff normalization lost original region block")
        block = json.loads(row[0])
        store_type._verify_metadata(block, identity)
        if block.get("proof_rebinding") is not None:
            from .proof_maintenance import verify_rebinding
            if identity in visited:
                raise ValueError("Cutoff normalization contains a region reference cycle")
            verify_rebinding(store_type, connection, block["proof_rebinding"],
                MaterializationKey.from_dict(block["key"]), block["execution_policy"], visited=(*visited, identity))
        rows = connection.execute("SELECT content_identity,payload_json,payload_identity FROM region_evidence "
            "WHERE identity=? AND kind='block'", (identity,)).fetchall()
        certificates = store_type._decode_region_evidence(block, "block", rows)
        certificate = next((value for value in certificates if set(seal["nodes"]) <= set(value["nodes"])), None)
        if certificate is None:
            raise ValueError("Cutoff normalization lost complete region block evidence")
        index = read(connection, block)
        if index is None or certificate["nodes"][manifest["node_id"]]["index"] != index:
            raise ValueError("Cutoff normalization region block differs from original coordinates")
        for node, key in seal["nodes"].items():
            if any(certificate["nodes"][node]["key"][field] != key[field] for field in ("node_id", "implementation_id")):
                raise ValueError("Cutoff normalization region block changed numerical definitions")
            member = certificate["nodes"][node]["index"]
            coordinates[node]["calendar"].extend(member["calendar"])
            coordinates[node]["membership"].update(member["membership"])
        if any(full["membership"].get(day) != index["membership"][day] for day in index["calendar"]):
            raise ValueError("Cutoff normalization region block changed membership")
        calendar.extend(index["calendar"])
    if calendar != full["calendar"]:
        raise ValueError("Cutoff normalization region blocks do not cover the complete interval")
    return {node: hash_mapping(value) for node, value in coordinates.items()}


def apply_normalization(store, plan, *, check_canceled=lambda: None):
    """Publish the admitted durable closure by reference, preserving all originals."""
    from .incremental import _reference_node
    results = {}
    for node in LogicalGraphSpec.from_dict(plan["spec"]).nodes:
        item = plan["nodes"].get(node.node_id, {})
        if "original_identity" not in item:
            continue
        check_canceled()
        key = MaterializationKey.from_dict(item["replacement_key"])
        with store.read_transaction() as connection:
            verify_normalization(type(store), connection, plan, key, item["policy"])
        if store.describe(key.identity) is None:
            source = store.lookup(MaterializationKey.from_dict(item["original_key"]), evidence=False)
            value = _reference_node(source.panel.lazy(include_traces=True), source.panel.domain, key,
                value_type=source.panel.value_type, trace_columns=source.panel.trace_columns)
            store.publish(NodeMaterialization(key, value), execution_policy=item["policy"],
                value_references=(source.key.identity,), _proof_rebinding=plan,
                region_evidence=None if item["replacement_seal"] is None else {"seal": item["replacement_seal"]})
        results[node.node_id] = key.identity
        check_canceled()
    return {"outcome": "rebound", "original_identity": plan["original_identity"],
        "results": results, "coverage": plan["coverage"], "computed_sessions": 0}
