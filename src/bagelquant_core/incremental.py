"""Canonical finite-history blocks, independent of application scheduling.

Every first build and rebuild uses the same 32-session mathematical blocks.
Block reuse is proven from all parent values, coordinates, membership and traces.
Unbounded/custom contracts conservatively replay their required full history.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Mapping

import numpy as np
import polars as pl

from .domain import Domain
from .execution import ExecutionRuntime
from .graph import Graph
from .graph_management import parents
from .hashing import hash_dataframe, hash_mapping
from .logical import LogicalGraphSpec
from .materialization import MaterializationKey, NodeMaterialization, materialization_trace_identity
from .node import Node
from .operator import OPERATOR_REGISTRY

CANONICAL_SESSIONS = 32
FINITE_EXECUTION_POLICY = "finite_causal_blocks.v2"


def finite_execution_policies(spec, *, anchor):
    policies = {}
    for node in spec.nodes:
        if node.node_type != "operator":
            continue
        registered = OPERATOR_REGISTRY.get(node.operator)
        history = registered.history(node.parameters)
        if history is not None:
            policies[node.node_id] = {"name": FINITE_EXECUTION_POLICY,
                "sessions": CANONICAL_SESSIONS, "anchor": str(anchor),
                "history_observations": history}
    return policies


def _canonical_halos(domain, observations):
    calendar = domain.times.cast(pl.Int32).to_numpy()
    if not len(calendar):
        return {}
    origins = np.arange(0, len(calendar), CANONICAL_SESSIONS)
    earliest = calendar[np.maximum(0, origins-observations)].copy()
    if observations and domain.is_dynamic:
        membership = domain.grid_lazy().sort("asset_id", "time").collect()
        times = membership["time"].cast(pl.Int32).to_numpy()
        lengths = membership["asset_id"].rle().struct.field("len").to_numpy()
        offset = 0
        for length in lengths:
            asset_times = times[offset:offset+int(length)]
            offset += int(length)
            counts = np.searchsorted(asset_times, calendar[origins], side="left")
            present = counts > 0
            earliest[present] = np.minimum(earliest[present], asset_times[np.maximum(0, counts[present]-observations)])
    earliest[0] = calendar[0]
    epoch = date(1970, 1, 1)
    return {int(origin): epoch+timedelta(days=int(day)) for origin, day in zip(origins, earliest, strict=True)}


def scoped_domain(domain, start, end):
    dates = domain.times.filter(domain.times.is_between(start, end))
    universe = (domain.grid_lazy().filter(pl.col("time").is_between(start, end)).collect().with_columns(
        pl.lit(True).alias("active")) if domain.is_dynamic else domain.asset_ids)
    return Domain(calendar=dates, universe=universe)


def execute_finite_node(spec, node_id, *, inputs: Mapping[str, Node], completed: Mapping[str, Node],
                        parameters, key, domain, policy, store, check_canceled):
    """Execute one admitted node, verifying content proofs before reusing blocks."""
    declared = {node.node_id: node for node in spec.nodes}
    node = declared[node_id]
    direct = {parent: (inputs[declared[parent].input_key] if declared[parent].node_type == "input"
                       else completed[parent]) for parent in parents(node)}
    calendar = domain.times.to_list()
    if not calendar:
        raise ValueError("finite update requires an explicit nonempty calendar")
    if date.fromisoformat(policy["anchor"]) > calendar[0]:
        raise ValueError("block anchor must precede the first observation session")
    # Membership halos count active coordinates, including long universe gaps.
    halos = _canonical_halos(domain, policy["history_observations"])
    local = LogicalGraphSpec({node_id: node_id}, spec.nodes)
    blocks = []
    computed = reused = 0
    for begin in range(0, len(calendar), CANONICAL_SESSIONS):
        check_canceled()
        last = calendar[min(begin+CANONICAL_SESSIONS, len(calendar))-1]
        warm = halos[begin]
        scoped = {}
        proofs = []
        for parent, value in direct.items():
            frame = value.lazy(include_traces=True).filter(pl.col("time").is_between(warm, last)).collect().sort("time", "asset_id")
            own = scoped_domain(value.domain, warm, last)
            proof = hash_mapping({"frame": hash_dataframe(frame), "domain": own.signature,
                "type": value.value_type.value, "traces": value.trace_columns})
            scoped[parent] = Node.from_domain(frame, own, identity=proof, value_type=value.value_type,
                trace_columns=value.trace_columns, trace_identity=proof if value.trace_columns else None)
            proofs.append(proof)
        output_domain = scoped_domain(domain, calendar[begin], last)
        block_key = MaterializationKey(node_id, key.implementation_id, tuple(proofs), output_domain.signature,
            hash_mapping({"block": policy, "start": str(calendar[begin]), "end": str(last), "parameters": parameters}))
        record = store.lookup(block_key)
        if record is None:
            bindings = {declared[parent].input_key: value for parent, value in scoped.items() if declared[parent].node_type == "input"}
            retained = {parent: value for parent, value in scoped.items() if declared[parent].node_type == "operator"}
            bound = Graph.from_logical_spec(local, inputs=bindings, node_bindings=retained, parameter_bindings=parameters)
            runtime = ExecutionRuntime(check_canceled=check_canceled)
            result = runtime.run(bound, dense_output=False)
            if runtime.node_artifacts or runtime.node_checkpoints or runtime.node_training_audits:
                raise ValueError("finite contract cannot discard operator evidence")
            frame = result.collect(dense=False, include_traces=True).filter(pl.col("time") >= calendar[begin])
            value = Node.from_domain(frame, output_domain, value_type=result.value_type,
                identity=block_key.identity, trace_columns=result.trace_columns,
                trace_identity=materialization_trace_identity(block_key, result.trace_columns))
            check_canceled()
            store.publish(NodeMaterialization(block_key, value), execution_policy=policy)
            record = store.lookup(block_key)
            computed += 1
        else:
            reused += 1
        blocks.append(record.panel)
    result = Node.from_domain(pl.concat([value.lazy(include_traces=True) for value in blocks]), domain,
        identity=key.identity, value_type=blocks[0].value_type, trace_columns=blocks[0].trace_columns,
        trace_identity=materialization_trace_identity(key, blocks[0].trace_columns))
    check_canceled()
    store.publish(NodeMaterialization(key, result), execution_policy=policy)
    return store.lookup(key), {"computed_blocks": computed, "reused_blocks": reused, "canonical_sessions": CANONICAL_SESSIONS}


def execute_checkpoint_node(spec, node_id, *, inputs, completed, parameters, key, domain, store, check_canceled):
    """Restore only when every direct parent's complete prefix is byte-proven.

    The full observation calendar remains available to training windows and
    trace propagation. A checkpoint skips completed numerical state transitions;
    it never changes the frozen information cutoff or discards historical proof.
    """
    from .operator_state import capture_operator_checkpoints
    declared = {node.node_id: node for node in spec.nodes}
    direct = {parent: (inputs[declared[parent].input_key] if declared[parent].node_type == "input" else completed[parent])
              for parent in parents(declared[node_id])}
    def proofs(end):
        result = {}
        for parent, value in direct.items():
            selected = value.lazy(include_traces=True).filter(pl.col("time") <= end).collect().sort("time", "asset_id")
            calendar = value.domain.times.filter(value.domain.times <= end)
            membership = value.domain.grid_lazy().filter(pl.col("time") <= end).collect().sort("time", "asset_id")
            result[parent] = hash_mapping({"values": hash_dataframe(selected), "calendar": hash_dataframe(pl.DataFrame({"time": calendar})),
                "membership": hash_dataframe(membership), "type": value.value_type.value, "traces": value.trace_columns})
        return result
    policy = {"name": "checkpoint_prefix.v1", "parameters": parameters, "parent_proofs": proofs(domain.times.max())}
    previous = None
    manifests = []
    for item in store.inventory():
        if item["node_id"] == node_id and item["key"]["implementation_id"] == key.implementation_id:
            manifest = store.manifest(item["identity"])
            prior_policy = manifest.get("execution_policy") or {}
            if (prior_policy.get("name") == policy["name"] and prior_policy.get("parameters") == parameters
                    and manifest["coverage_start"] == str(domain.times.min())
                    and manifest["through"] < str(domain.times.max()) and manifest.get("checkpoint")):
                manifests.append(manifest)
    for manifest in sorted(manifests, key=lambda item: item["through"], reverse=True):
        check_canceled()
        end = date.fromisoformat(manifest["through"])
        if manifest["execution_policy"]["parent_proofs"] != proofs(end):
            continue
        try:
            candidate = store.read(manifest["identity"])
        except (ValueError, OSError):
            # A damaged optional checkpoint does not invalidate new source evidence.
            continue
        if candidate.checkpoint.get("state", {}).get("through") == str(end):
            previous = candidate
            break
    local = LogicalGraphSpec({node_id: node_id}, spec.nodes)
    bound = Graph.from_logical_spec(local, inputs=inputs, node_bindings=completed, parameter_bindings=parameters)
    runtime = ExecutionRuntime(check_canceled=check_canceled)
    restored = {} if previous is None else {node_id: previous.checkpoint}
    with capture_operator_checkpoints(restored, calendar=domain.times.to_list()) as checkpoints:
        result = runtime.run(bound, dense_output=False)
        frame = result.collect(dense=False, include_traces=True)
    artifacts = dict(runtime.node_artifacts.get(node_id, {}))
    audits = tuple(runtime.node_training_audits.get(node_id, ()))
    through = None
    if previous is not None:
        through = date.fromisoformat(previous.checkpoint["state"]["through"])
        frame = pl.concat([previous.panel.collect(dense=False, include_traces=True), frame.filter(pl.col("time") > through)]).sort("time", "asset_id")
        audits = (*previous.training_audits, *audits)
        for name in set(previous.artifacts) | set(artifacts):
            old, new = previous.artifacts.get(name), artifacts.get(name)
            if isinstance(old, pl.DataFrame) and isinstance(new, pl.DataFrame) and "time" in old.columns:
                artifacts[name] = pl.concat([old.filter(pl.col("time") <= through), new.filter(pl.col("time") > through)]).sort("time")
            elif name not in artifacts:
                artifacts[name] = old
    value = Node.from_domain(frame, domain, identity=key.identity, value_type=result.value_type,
        trace_columns=result.trace_columns, trace_identity=materialization_trace_identity(key, result.trace_columns))
    record = NodeMaterialization(key, value, artifacts=artifacts, checkpoint=checkpoints.captured.get(node_id), training_audits=audits)
    check_canceled()
    store.publish(record, execution_policy=policy)
    return store.read(key.identity), {"checkpoint_through": None if through is None else str(through),
        "replayed_sessions": len(domain.times) if through is None else len(domain.times.filter(domain.times > through))}
