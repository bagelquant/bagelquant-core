"""Frozen dependency plans exposing caller-admitted node execution."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from threading import RLock
from uuid import uuid4
from typing import Mapping

from .execution import ExecutionRuntime
from .graph_management import parents
from .logical import LogicalGraphSpec
from .materialization import MaterializationStatus
from .resources import ResourceLimits, resource_limits
from .store import RevisionConflict
from .incremental import finite_execution_policies, execute_finite_node, execute_checkpoint_node
from .hashing import hash_mapping


class UpdatePlan:
    """One frozen global or branch update, with no worker/admission policy."""

    def __init__(self, graph, contexts: Mapping, *, through, roots=None):
        from .graph import Graph
        if graph._store is None:
            raise ValueError("persistent updates require an explicit CoreStore")
        self.request_id = uuid4().hex
        self.graph = graph
        self.store = graph._store
        self.state = graph._state()
        self.mode = "global" if roots is None else "branch"
        self.through = date.fromisoformat(str(through)).isoformat()
        self._lock = RLock()
        self._in_flight = set()
        self.results = {context: {} for context in contexts}
        self._values = {context: {} for context in contexts}
        if not contexts:
            raise ValueError("an update requires at least one execution context")
        if self.mode == "global" and set(contexts) != set(self.state["contexts"]):
            raise ValueError("global update requires every registered context")
        if roots is None:
            selected = {node.node_id for node in self.state["spec"].nodes
                        if self.state["statuses"][node.node_id] == "active"}
        else:
            roots = (roots,) if isinstance(roots, str) else roots
            selected = set()
            for root in roots:
                selected.update(graph.upstream(root, include_self=True))
            if any(self.state["statuses"][node_id] != "active" for node_id in selected):
                raise ValueError("branch update contains sleeping nodes")
        self.spec = LogicalGraphSpec({node_id: node_id for node_id in sorted(selected)},
            tuple(node for node in self.state["spec"].nodes if node.node_id in selected))
        self._nodes = {node.node_id: node for node in self.spec.nodes if node.node_type == "operator"}
        self._inputs = {}
        self._parameters = {}
        self.keys = {}
        self.domains = {}
        self.policies = {}
        self.numerical_contexts = {}
        self.resource_usage = {context: {} for context in contexts}
        for context_id, bindings in contexts.items():
            needed = {node.input_key for node in self.spec.nodes if node.node_type == "input"}
            for source in needed:
                if source not in bindings:
                    raise ValueError(f"context {context_id!r} is blocked: missing source {source!r}")
                value = bindings[source]
                if not getattr(value, "_durable_identity", False):
                    raise ValueError("persistent updates require immutable source evidence for lazy inputs")
                if not len(value.domain.times) or str(value.domain.times.max()) != self.through:
                    raise ValueError(f"context {context_id!r} source {source!r} does not cover the requested final session")
            self._inputs[context_id] = dict(bindings)
            self._parameters[context_id] = self.state["contexts"].get(context_id, {}).get("parameters", {})
            bound = Graph.from_logical_spec(self.spec, inputs=bindings, parameter_bindings=self._parameters[context_id])
            definition = self.state["contexts"].get(context_id)
            if definition is None:
                raise ValueError("execution context must be explicitly registered")
            self.policies[context_id] = finite_execution_policies(self.spec, anchor=definition["anchor"])
            self.numerical_contexts[context_id] = {node_id: hash_mapping({"history_policy": self.policies[context_id].get(node_id),
                "information_cutoff": definition.get("information_cutoff")}) for node_id in self._nodes}
            planner = ExecutionRuntime(node_contexts=self.numerical_contexts[context_id])
            self.keys[context_id] = planner.plan_materialization_keys(bound)
            domains = {node.logical_id: node.domain for node in bound.nodes if node.node_type == "input"}
            domains.update(planner.node_domains)
            unsupported = {node_id for node_id in self.policies[context_id]
                if str(domains[node_id].times.min()) != str(definition["anchor"]) or any(not domains[parent].times.equals(domains[node_id].times) for parent in parents(self._nodes[node_id]))}
            if unsupported:
                self.policies[context_id] = {node: policy for node, policy in self.policies[context_id].items() if node not in unsupported}
                self.numerical_contexts[context_id] = {node_id: hash_mapping({"history_policy": self.policies[context_id].get(node_id),
                    "information_cutoff": definition.get("information_cutoff")}) for node_id in self._nodes}
                planner = ExecutionRuntime(node_contexts=self.numerical_contexts[context_id])
                self.keys[context_id] = planner.plan_materialization_keys(bound)
            self.domains[context_id] = dict(planner.node_domains)
        self.guard()

    def guard(self):
        self.store.assert_revision(self.graph.graph_id, self.state["revision"], self.state["state_revision"])

    def ready(self, context_id):
        """Return a stable frontier; caller chooses how many nodes to admit."""
        self.guard()
        with self._lock:
            complete = self.results[context_id]
            return tuple(node_id for node_id, node in self._nodes.items()
                if node_id not in complete and (context_id, node_id) not in self._in_flight
                and all(parent not in self._nodes or parent in complete for parent in parents(node)))

    def execute(self, context_id, node_id, *, limits: ResourceLimits | None = None, check_canceled=None):
        """Execute one ready node, reusing already completed immutable parents."""
        from .graph import Graph
        def guard():
            if check_canceled is not None:
                check_canceled()
            self.guard()
        with self._lock:
            if node_id not in self.ready(context_id):
                raise ValueError("node is not ready or already admitted")
            self._in_flight.add((context_id, node_id))
        try:
            guard()
            key = self.keys[context_id][node_id]
            lookup = self.store.query(key)
            if lookup.status == MaterializationStatus.HIT:
                result = lookup.materialization
            elif node_id in self.policies[context_id]:
                with resource_limits(replace(limits or ResourceLimits(), parallel_nodes=1)):
                    result, usage = execute_finite_node(self.spec, node_id, inputs=self._inputs[context_id],
                        completed=self._values[context_id], parameters=self._parameters[context_id], key=key,
                        domain=self.domains[context_id][node_id], policy=self.policies[context_id][node_id],
                        store=self.store, check_canceled=guard)
                self.resource_usage[context_id][node_id] = usage
            elif self._checkpoint_supported(node_id):
                with resource_limits(replace(limits or ResourceLimits(), parallel_nodes=1)):
                    result, usage = execute_checkpoint_node(self.spec, node_id, inputs=self._inputs[context_id],
                        completed=self._values[context_id], parameters=self._parameters[context_id], key=key,
                        domain=self.domains[context_id][node_id], store=self.store, check_canceled=guard)
                self.resource_usage[context_id][node_id] = usage
            else:
                local = LogicalGraphSpec({node_id: node_id}, self.spec.nodes)
                bound = Graph.from_logical_spec(local, inputs=self._inputs[context_id],
                    parameter_bindings=self._parameters[context_id], node_bindings=self._values[context_id])
                runtime = ExecutionRuntime(materialization_store=self.store, check_canceled=guard,
                    node_contexts=self.numerical_contexts[context_id])
                with resource_limits(replace(limits or ResourceLimits(), parallel_nodes=1)):
                    runtime.run(bound, dense_output=False)
                result = runtime.node_materializations[node_id]
                if result.key != key:
                    raise ValueError("execution diverged from frozen materialization keys")
            guard()
            with self._lock:
                self.results[context_id][node_id] = result.key.identity
                self._values[context_id][node_id] = result.panel
            return result
        finally:
            with self._lock:
                self._in_flight.discard((context_id, node_id))

    def _checkpoint_supported(self, node_id):
        from .operator import OPERATOR_REGISTRY
        return OPERATOR_REGISTRY.get(self._nodes[node_id].operator).contract.checkpoint_replay

    @property
    def complete(self):
        with self._lock:
            return not self._in_flight and all(set(values) == set(self._nodes) for values in self.results.values())

    def publish(self, *, check_canceled=None):
        if check_canceled is not None:
            check_canceled()
        self.guard()
        with self._lock:
            if not self.complete:
                raise ValueError("cannot publish an incomplete update")
            for context, values in self.results.items():
                if any(identity != self.keys[context][node].identity for node, identity in values.items()):
                    raise RevisionConflict("update result identities changed")
            return self.store.commit_update(self.graph.graph_id, revision=self.state["revision"],
                state_revision=self.state["state_revision"], expected_previous=self.state["current_update"],
                results=self.results, through=self.through, mode=self.mode, request_id=self.request_id)
