"""Bounded causal-version replay in the ordinary Core graph/update store.

Sources supply typed scalar versions and opaque interval evidence. Core knows
neither their provider nor their storage. Earlier observations newly visible
on a later day affect that day's trailing calculations, never preceding
published outputs. Ordinary contemporaneous appends are evaluated in batches.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from .domain import Domain
from .execution import ExecutionRuntime
from .graph import Graph
from .graph_management import parents
from .hashing import hash_mapping
from .incremental import CANONICAL_SESSIONS, scoped_domain, _reference_node, _relevant_parameters
from .logical import LogicalGraphSpec
from .materialization import MaterializationKey, NodeMaterialization, materialization_trace_identity
from .node import Node
from .operation_contract import causal_history_requirements
from .operator import OPERATOR_REGISTRY
from .resources import active_resource_limits

CAUSAL_EXECUTION_POLICY = "causal_version_blocks.v2"


def _calendar_start(domain, start, history):
    dates = domain.times.to_list()
    first = next((index for index, day in enumerate(dates) if day >= start), len(dates))
    return dates[max(0, first-history)]


@dataclass(frozen=True, slots=True)
class CausalInputWindow:
    """One frozen bounded scalar-version acquisition.

    ``known_at`` is the source's effective visibility date. ``revision`` is
    the source's deterministic precedence for versions sharing a natural key;
    it must preserve the original source selection order, including ties.
    Null values are explicit versions, including withdrawals, not missing rows.
    """
    frame: pl.DataFrame
    source_identity: str

    def __post_init__(self):
        if not isinstance(self.frame, pl.DataFrame) or not isinstance(self.source_identity, str) or not self.source_identity:
            raise ValueError("causal versions require a typed frame and immutable source identity")
        required = {"time": pl.Date, "asset_id": pl.String, "known_at": pl.Date, "revision": pl.UInt64}
        if "value" not in self.frame.columns or any(self.frame.schema.get(name) != dtype for name, dtype in required.items()):
            raise TypeError("causal versions require time/known_at Date, asset_id String, revision UInt64 and value")
        if any(self.frame[name].null_count() for name in required):
            raise ValueError("causal natural keys and precedence cannot be null")
        if self.frame.unique(["time", "asset_id", "known_at", "revision"]).height != self.frame.height:
            raise ValueError("causal version precedence must uniquely identify versions")


def causal_nodes(spec, input_keys):
    affected = set()
    for node in spec.nodes:
        if node.node_type == "input" and node.input_key in input_keys or any(parent in affected for parent in parents(node)):
            affected.add(node.node_id)
    return {node.node_id for node in spec.nodes if node.node_type == "operator" and node.node_id in affected}


class CausalReplay:
    """One plan's coalesced causal work, shared by its logical graph nodes."""

    def __init__(self, plan, context):
        self.plan, self.context = plan, context
        self.nodes = {node.node_id: node for node in plan.spec.nodes}
        self.history = causal_history_requirements(plan.spec)
        self.selected = tuple(node.node_id for node in plan.spec.nodes if node.node_id in plan._causal_nodes)
        if any(self.history[node] is None for node in self.selected):
            raise ValueError("causal-version replay requires declared finite backward history for every node")
        if plan._input_scope_proof is None or plan._input_version_loader is None:
            raise ValueError("causal-version replay requires bounded version loading and scoped metadata proofs")
        self.completed = False
        self.usage = {}
        self.buffers = {}
        self.loaded_through = {}
        self.loaded_from = {}
        self.closures = {node.node_id: self._closure(node.node_id) for node in plan.spec.nodes if node.node_type == "operator"}
        self.traces = {}
        for node in plan.spec.nodes:
            self.traces[node.node_id] = (plan._inputs[context][node.input_key].trace_columns if node.node_type == "input"
                else tuple(dict.fromkeys(trace for parent in parents(node) for trace in self.traces[parent])))

    def _closure(self, node):
        result, pending = set(), [node]
        while pending:
            current = pending.pop()
            if current not in result:
                result.add(current)
                pending.extend(parents(self.nodes[current]))
        return tuple(item.node_id for item in self.plan.spec.nodes if item.node_id in result)

    def _policy(self, node):
        closure = self.closures[node]
        parameters = self.plan._parameters[self.context]
        return {"name": CAUSAL_EXECUTION_POLICY, "sessions": CANONICAL_SESSIONS,
            "anchor": self.plan.state["contexts"][self.context]["anchor"],
            **({"input_definitions": self.plan._causal_definitions[self.context][node]}
                if self.plan._causal_definitions[self.context][node] else {}),
            "closure": hash_mapping({"nodes": [self.nodes[key].to_dict() for key in closure],
                "implementations": {key: self.plan.keys[self.context][key].implementation_id
                    for key in closure if self.nodes[key].node_type == "operator"},
                "parameters": {key: _relevant_parameters(self.nodes[key], parameters) for key in closure}})}

    def _key(self, node, first, last):
        domain = self.plan.domains[self.context][node]
        warm = _calendar_start(domain, first, self.history[node])
        tokens = []
        for key in self.closures[node]:
            source = self.nodes[key]
            if source.node_type != "input":
                continue
            value = self.plan._inputs[self.context][source.input_key]
            token = self.plan._input_scope_proof(self.context, source.input_key, warm, last, last)
            if not isinstance(token, str) or not token:
                raise ValueError("causal update needs scoped immutable input evidence before numerical work")
            # Source coverage can acquire a later-discovered old key. Its
            # output membership and as-of token, not future source coverage,
            # determine an already published causal block's authority.
            tokens.append(hash_mapping({"source": source.input_key, "content": token,
                "type": value.value_type.value, "traces": value.trace_columns}))
        own = scoped_domain(domain, first, last)
        return MaterializationKey(node, self.plan.keys[self.context][node].implementation_id,
            tuple(tokens), own.signature, hash_mapping({"policy": self._policy(node),
                "start": str(first), "end": str(last)})), own

    def proof_windows(self):
        for node in self.selected:
            domain = self.plan.domains[self.context][node]
            calendar = domain.times.to_list()
            for begin in range(0, len(calendar), CANONICAL_SESSIONS):
                first, last = calendar[begin], calendar[min(begin + CANONICAL_SESSIONS, len(calendar))-1]
                warm = _calendar_start(domain, first, self.history[node])
                for key in self.closures[node]:
                    source = self.nodes[key]
                    if source.node_type == "input":
                        yield node, source.input_key, warm, last, last

    def execute(self, requested, guard):
        if not self.completed:
            self._run(guard)
            self.completed = True
        result = self.plan.store.lookup(self.plan.keys[self.context][requested],
            domain=self.plan.domains[self.context][requested], evidence=False)
        if result is None:
            raise ValueError("causal result publication is missing")
        return result, self.usage[requested]

    def _run(self, guard):
        plan, context = self.plan, self.context
        calendar = plan.domains[context][self.selected[0]].times.to_list()
        if any(plan.domains[context][node].times.to_list() != calendar for node in self.selected):
            raise ValueError("shared causal replay requires one explicitly aligned calendar")
        references = {node: [] for node in self.selected}
        self.usage = {node: {"computed_blocks": 0, "reused_blocks": 0,
            "computed_output_sessions": 0, "kernel_sessions": 0, "input_sessions": 0,
            "causal_segments": 0, "canonical_sessions": CANONICAL_SESSIONS} for node in self.selected}
        inventory = {node: [plan.store.manifest(item["identity"]) for item in plan.store.inventory(
            node_id=node, implementation_id=plan.keys[context][node].implementation_id)
            if plan.store.manifest(item["identity"]).get("execution_policy") == self._policy(node)] for node in self.selected}
        # A new opaque proof namespace must not silently cold-replay an entire
        # completed model. Real bounded repairs retain at least one proven
        # interval; definitions/implementations changing create new authority.
        for node in self.selected:
            completed = plan.store.completed_materializations(plan.graph.graph_id, context, node)
            if not completed:
                continue
            candidates = inventory[node]
            matching_authority = any(plan.store.describe(identity)["key"]["implementation_id"] ==
                plan.keys[context][node].implementation_id and plan.store.manifest(identity).get("execution_policy") == self._policy(node)
                for identity in completed)
            if matching_authority and not any(self._key(node, date.fromisoformat(item["coverage_start"]),
                    date.fromisoformat(item["through"]))[0].identity == item["identity"]
                    for item in candidates if item["through"] <= plan.through):
                raise ValueError("Completed causal history has no proven retained interval; "
                    "plan affected intervals or perform explicit input-proof maintenance")
        for begin in range(0, len(calendar), CANONICAL_SESSIONS):
            guard()
            first, last = calendar[begin], calendar[min(begin + CANONICAL_SESSIONS, len(calendar))-1]
            missing, block_keys, prefixes = {}, {}, {}
            for node in self.selected:
                key, own = self._key(node, first, last)
                block_keys[node] = key
                if plan.store.describe(key.identity) is not None:
                    references[node].append(key.identity)
                    self.usage[node]["reused_blocks"] += 1
                    continue
                missing[node] = first
                for candidate in sorted(inventory[node], key=lambda row: row["through"], reverse=True):
                    end = date.fromisoformat(candidate["through"])
                    if candidate["coverage_start"] != str(first) or not first <= end < last or end not in calendar:
                        continue
                    prior_key, _ = self._key(node, first, end)
                    if prior_key.identity == candidate["identity"]:
                        prefixes[node] = candidate["identity"]
                        missing[node] = calendar[calendar.index(end)+1]
                        break
            if not missing:
                continue
            output_first = min(missing.values())
            windows = self._required_windows(missing, last)
            acquired = self._acquire(windows, last, guard)
            boundaries = {output_first}
            for frame in acquired.values():
                boundaries.update(frame.filter((pl.col("known_at") > pl.col("time")) &
                    pl.col("known_at").is_between(output_first, last))["known_at"].to_list())
            segments = sorted(boundaries)
            parts = {node: [] for node in missing}
            for index, segment_first in enumerate(segments):
                segment_last = segments[index+1] - timedelta(days=1) if index+1 < len(segments) else last
                admitted = {node: max(start, segment_first) for node, start in missing.items() if start <= segment_last}
                if not admitted:
                    continue
                guard()
                results = self._evaluate(acquired, admitted, segment_last, guard)
                for node, value in results.items():
                    parts[node].append(value)
                    self.usage[node]["causal_segments"] += 1
            for node, pieces in parts.items():
                guard()
                key = block_keys[node]
                own = scoped_domain(plan.domains[context][node], missing[node], last)
                frame = pl.concat(pieces).sort("time", "asset_id")
                fragment = key if node not in prefixes else MaterializationKey(node, key.implementation_id,
                    key.inputs, own.signature, hash_mapping({"fragment_of": key.identity, "start": str(missing[node])}))
                value = Node.from_domain(frame, own, identity=fragment.identity,
                    value_type=plan._bound_nodes[context][node].value_type,
                    trace_columns=self.traces[node],
                    trace_identity=materialization_trace_identity(fragment, self.traces[node]))
                plan.store.publish(NodeMaterialization(fragment, value), execution_policy=self._policy(node))
                if node in prefixes:
                    old = plan.store.lookup(MaterializationKey.from_dict(plan.store.describe(prefixes[node])["key"]),
                        domain=scoped_domain(plan.domains[context][node], first, calendar[calendar.index(missing[node])-1]), evidence=False)
                    panel = _reference_node(pl.concat([old.panel.lazy(include_traces=True), value.lazy(include_traces=True)]),
                        scoped_domain(plan.domains[context][node], first, last), key,
                        value_type=value.value_type, trace_columns=value.trace_columns)
                    plan.store.publish(NodeMaterialization(key, panel), execution_policy=self._policy(node),
                        value_references=(prefixes[node], fragment.identity))
                references[node].append(key.identity)
                self.usage[node]["computed_blocks"] += 1
                self.usage[node]["computed_output_sessions"] += len(own.times)
        for node in self.selected:
            guard()
            key, domain = plan.keys[context][node], plan.domains[context][node]
            frames = [plan.store.lookup(MaterializationKey.from_dict(plan.store.describe(identity)["key"]),
                domain=domain.window(date.fromisoformat(plan.store.describe(identity)["coverage_start"]),
                    date.fromisoformat(plan.store.describe(identity)["through"])), evidence=False).panel.lazy(include_traces=True)
                for identity in references[node]]
            value = _reference_node(pl.concat(frames), domain, key,
                value_type=plan._bound_nodes[context][node].value_type,
                trace_columns=self.traces[node])
            plan.store.publish(NodeMaterialization(key, value), execution_policy=self._policy(node), value_references=references[node])

    def _required_windows(self, requested, last):
        plan, context = self.plan, self.context
        starts = dict(requested)
        for node in reversed(plan.spec.nodes):
            if node.node_id not in starts or node.node_type == "input":
                continue
            history = OPERATOR_REGISTRY.get(node.operator).history(node.parameters)
            if history is None:
                raise ValueError("causal operation has no finite backward history contract")
            for parent in parents(node):
                domain = (plan._inputs[context][self.nodes[parent].input_key].domain
                    if self.nodes[parent].node_type == "input" else plan.domains[context][parent])
                warm = _calendar_start(domain, starts[node.node_id], history)
                starts[parent] = min(starts.get(parent, warm), warm)
        return starts

    def _acquire(self, windows, last, guard):
        plan, context = self.plan, self.context
        result = {}
        for key, warm in windows.items():
            source = self.nodes[key]
            if source.node_type != "input":
                continue
            guard()
            expected = plan._inputs[context][source.input_key]
            if source.input_key in plan._causal_inputs:
                cutoff = date.fromisoformat(plan.state["contexts"][context].get("information_cutoff") or plan.through)
                def load(first, end):
                    guard()
                    window = plan._input_version_loader(context, source.input_key, first, end, cutoff)
                    if not isinstance(window, CausalInputWindow) or window.source_identity != expected.identity:
                        raise ValueError("causal input differs from its frozen source authority")
                    frame = window.frame
                    wanted = pl.String if expected.value_type == "category" else pl.Float64
                    if frame.schema["value"] != wanted or set(expected.trace_columns) - set(frame.columns):
                        raise TypeError("causal input differs from its declared typed channels")
                    if frame.filter(~pl.col("time").is_between(first, end) | (pl.col("known_at") > cutoff)).height:
                        raise ValueError("causal input widened its frozen window or visibility cutoff")
                    if not frame.select("time", "asset_id").unique().join(
                            expected.domain.window(first, end).grid_lazy().collect(), on=["time", "asset_id"], how="anti").is_empty():
                        raise ValueError("causal input contains coordinates outside the frozen Domain")
                    return frame.select(*dict.fromkeys(("time", "asset_id", "value", "known_at", "revision", *expected.trace_columns)))
                previous = self.loaded_through.get(key)
                if previous is None:
                    self.buffers[key] = load(warm, last)
                    self.loaded_from[key] = warm
                else:
                    frames = [self.buffers[key]]
                    if warm < self.loaded_from[key]:
                        frames.insert(0, load(warm, self.loaded_from[key]-timedelta(days=1)))
                    if previous < last:
                        frames.append(load(max(warm, previous+timedelta(days=1)), last))
                    self.buffers[key] = pl.concat(frames)
                self.loaded_through[key] = last
                calendar = plan.domains[context][self.selected[0]].times.to_list()
                block_first = calendar[(calendar.index(last)//CANONICAL_SESSIONS)*CANONICAL_SESSIONS]
                retained_from = min(warm, _calendar_start(expected.domain, block_first,
                    max(self.history[node] or 0 for node in self.selected)))
                self.loaded_from[key] = max(self.loaded_from[key], retained_from)
                self.buffers[key] = self.buffers[key].filter(pl.col("time") >= self.loaded_from[key])
                result[key] = self.buffers[key].filter(pl.col("time").is_between(warm, last))
            else:
                value = plan._window_reader(context, guard, None)(key, expected, warm, last) if plan._input_window_loader else expected
                result[key] = value.collect(dense=False, include_traces=True).filter(pl.col("time").is_between(warm, last)).with_columns(
                    pl.col("time").alias("known_at"), pl.lit(0, dtype=pl.UInt64).alias("revision"))
            for node in self.selected:
                if key in self.closures[node]:
                    self.usage[node]["input_sessions"] += len(expected.domain.window(warm, last).times)
            if sum(frame.estimated_size() for frame in self.buffers.values()) > active_resource_limits().memory_target_mib * 1024 * 1024 // 4:
                raise MemoryError("causal source window exceeds the admitted bounded memory reserve")
        return result

    def _evaluate(self, acquired, requested, last, guard):
        plan, context = self.plan, self.context
        starts = self._required_windows(requested, last)
        values = {}
        membership = plan.domains[context][self.selected[0]].window(min(starts.values()), last)
        assets = sorted(set(membership.asset_ids.to_list()).union(*(set(frame["asset_id"].to_list())
            for frame in acquired.values())))
        for node in plan.spec.nodes:
            if node.node_id not in starts:
                continue
            guard()
            original_domain = plan._inputs[context][node.input_key].domain if node.node_type == "input" else plan.domains[context][node.node_id]
            # Output membership is causal observed coordinates; temporary
            # rolling windows retain null sessions between observations.
            own = Domain(calendar=original_domain.times.filter(original_domain.times.is_between(starts[node.node_id], last)), universe=assets)
            if not len(own.times):
                continue
            if node.node_type == "input":
                expected = plan._inputs[context][node.input_key]
                frame = acquired[node.node_id].filter((pl.col("known_at") <= last) &
                    pl.col("time").is_between(starts[node.node_id], last)).sort("known_at", "revision").unique(
                        ["time", "asset_id"], keep="last", maintain_order=True).select("time", "asset_id", "value", *expected.trace_columns)
                values[node.node_id] = Node.from_domain(frame, own, identity=expected.identity,
                    value_type=expected.value_type, trace_columns=expected.trace_columns, trace_identity=expected.trace_identity)
                continue
            history = OPERATOR_REGISTRY.get(node.operator).history(node.parameters)
            if history is None:
                raise ValueError("causal operation has no finite backward history contract")
            warm = _calendar_start(original_domain, starts[node.node_id], history)
            work = Domain(calendar=original_domain.times.filter(original_domain.times.is_between(warm, last)),
                universe=assets)
            # A shared parent can also serve a longer downstream halo. Bind
            # every peer to this operation's exact consumed calendar before
            # evaluating, retaining the already computed parent values.
            direct = {key: Node.from_domain(values[key].lazy(include_traces=True).filter(
                    pl.col("time").is_between(warm, last)), work,
                identity=values[key].identity, value_type=values[key].value_type,
                trace_columns=values[key].trace_columns, trace_identity=values[key].trace_identity)
                for key in parents(node)}
            inputs = {self.nodes[key].input_key: value for key, value in direct.items() if self.nodes[key].node_type == "input"}
            retained = {key: value for key, value in direct.items() if self.nodes[key].node_type == "operator"}
            local = LogicalGraphSpec({node.node_id: node.node_id}, tuple(item for item in plan.spec.nodes if item.node_id in self.closures[node.node_id]))
            graph = Graph.from_logical_spec(local, inputs=inputs, node_bindings=retained, parameter_bindings=plan._parameters[context])
            runtime = ExecutionRuntime(check_canceled=guard)
            result = runtime.run(graph, dense_output=False)
            if not isinstance(result, Node):
                raise ValueError("causal operation must return one typed Node")
            if runtime.node_artifacts or runtime.node_checkpoints or runtime.node_training_audits:
                raise ValueError("finite causal replay cannot discard numerical side channels")
            frame = result.collect(dense=False, include_traces=True).filter(pl.col("time").is_between(starts[node.node_id], last))
            values[node.node_id] = Node.from_domain(frame, own, identity="causal-work:" + hash_mapping(
                {"node": node.node_id, "start": str(starts[node.node_id]), "end": str(last), "sources": [value.identity for value in direct.values()]}),
                value_type=result.value_type, trace_columns=result.trace_columns,
                trace_identity=result.trace_identity)
            if node.node_id in self.usage:
                self.usage[node.node_id]["kernel_sessions"] += len(work.times)
        return {node: plan.domains[context][node].window(first, last).apply_membership_lazy(
                values[node].lazy(include_traces=True)).collect()
            for node, first in requested.items()}


__all__ = ["CAUSAL_EXECUTION_POLICY", "CausalInputWindow"]
