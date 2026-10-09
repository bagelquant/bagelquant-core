"""Graph mutation and lifecycle mechanics, independent of scheduling policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .logical import LogicalGraphSpec


@dataclass(frozen=True)
class MergeResult:
    outputs: Mapping[str, str]
    node_map: Mapping[str, str]


def parents(node):
    return (*node.inputs, *(parent for values in node.panel_parameters.values() for parent in values))


class GraphManagement:
    def _state(self):
        if getattr(self, "_store", None) is not None:
            return self._store.graph_state(self.graph_id)
        spec = self.logical_spec()
        return {"revision": getattr(self, "_revision", 0), "state_revision": getattr(self, "_state_revision", 0),
                "spec": spec, "statuses": {node.node_id: getattr(self, "_statuses", {}).get(node.node_id, "active") for node in spec.nodes},
                "contexts": getattr(self, "_contexts", {}), "current_update": None}

    @staticmethod
    def _node_id(node):
        return node if isinstance(node, str) else node.logical_id

    def snapshot(self):
        """Read the graph definition, registered contexts, statuses and revisions."""
        return self._state()

    @property
    def revision(self):
        state = self._state()
        return state["revision"], state["state_revision"]

    def status(self, node):
        return self._state()["statuses"][self._node_id(node)]

    def upstream(self, node, *, include_self=False):
        state = self._state()
        declared = {item.node_id: item for item in state["spec"].nodes}
        node_id = self._node_id(node)
        if node_id not in declared:
            raise KeyError(node_id)
        found = {node_id} if include_self else set()
        pending = list(parents(declared[node_id]))
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(parents(declared[current]))
        return tuple(item.node_id for item in state["spec"].nodes if item.node_id in found)

    def downstream(self, node, *, include_self=False):
        state = self._state()
        node_id = self._node_id(node)
        if node_id not in state["statuses"]:
            raise KeyError(node_id)
        found = {node_id}
        for item in state["spec"].nodes:
            if any(parent in found for parent in parents(item)):
                found.add(item.node_id)
        if not include_self:
            found.remove(node_id)
        return tuple(item.node_id for item in state["spec"].nodes if item.node_id in found)

    def _save_state(self, state, spec, statuses, *, state_change=False):
        if getattr(self, "_store", None) is not None:
            self._store.save_graph(self.graph_id, spec, statuses,
                expected_revision=state["revision"], expected_state_revision=state["state_revision"], state_change=state_change)
        else:
            self._revision = state["revision"] + int(not state_change)
            self._state_revision = state["state_revision"] + int(state_change)
        self._logical_specification = spec
        self._statuses = statuses

    def merge(self, local):
        """Atomically intern a local graph; aliases never become node identities."""
        from .node import Node
        if isinstance(local, Node):
            local = local.graph
        incoming, node_map = local.logical_spec().normalized()
        state = self._state()
        blocked = [node.node_id for node in incoming.nodes if state["statuses"].get(node.node_id) == "sleeping"]
        if blocked:
            raise ValueError(f"cannot grow from sleeping nodes: {blocked}")
        existing = state["spec"]
        # Global outputs use stable IDs; local names are returned separately.
        left = LogicalGraphSpec({value: value for value in existing.outputs.values()}, existing.nodes)
        right = LogicalGraphSpec({value: value for value in incoming.outputs.values()}, incoming.nodes)
        union = left.union(right)
        statuses = {node.node_id: state["statuses"].get(node.node_id, "active") for node in union.nodes}
        if union.to_dict() != left.to_dict():
            self._save_state(state, union, statuses)
        return MergeResult(dict(incoming.outputs), node_map)

    def resolve_local(self, roots):
        """Resolve the full upstream closure without evaluating or reading values."""
        from .graph import Graph
        state = self._state()
        roots = (roots,) if isinstance(roots, str) else tuple(roots)
        selected = {self._node_id(root) for root in roots}
        required = set(selected)
        for root in selected:
            required.update(self.upstream(root))
        spec = LogicalGraphSpec({root: root for root in sorted(selected)}, tuple(node for node in state["spec"].nodes if node.node_id in required))
        return self._attach_guard(Graph.from_logical_spec(spec), state)

    def sleep(self, node):
        state = self._state()
        selected = self.downstream(node, include_self=True)
        statuses = dict(state["statuses"])
        for node_id in selected:
            statuses[node_id] = "sleeping"
        if statuses != state["statuses"]:
            self._save_state(state, state["spec"], statuses, state_change=True)
        return selected

    def wake(self, nodes):
        nodes = (nodes,) if isinstance(nodes, str) else tuple(nodes)
        selected = {self._node_id(node) for node in nodes}
        state = self._state()
        if not selected <= state["statuses"].keys():
            raise KeyError("unknown nodes in wake request")
        statuses = dict(state["statuses"])
        for node in state["spec"].nodes:
            if node.node_id in selected:
                if any(statuses[parent] != "active" for parent in parents(node)):
                    raise ValueError("cannot wake a node with sleeping upstream dependencies")
                statuses[node.node_id] = "active"
        if statuses != state["statuses"]:
            self._save_state(state, state["spec"], statuses, state_change=True)

    def bind(self, inputs, *, parameter_bindings=None):
        from .graph import Graph
        state = self._state()
        return self._attach_guard(Graph.from_logical_spec(state["spec"], inputs=inputs, parameter_bindings=parameter_bindings), state)

    def _attach_guard(self, bound, state):
        existing = getattr(self, "_execution_guards", ())
        selected = {node.node_id for node in bound.logical_spec().nodes}
        def guard():
            for check in existing:
                check()
            current = self._state()
            if (current["revision"], current["state_revision"]) != (state["revision"], state["state_revision"]):
                from .store import RevisionConflict
                raise RevisionConflict("graph/state changed after binding")
            if any(current["statuses"].get(node_id) != "active" for node_id in selected):
                raise ValueError("cannot execute or grow from sleeping nodes")
        bound._execution_guards = (guard,)
        for node in bound.nodes:
            node._execution_guards = (guard,)
        return bound

    @classmethod
    def from_dsl(cls, source, *, inputs, parameters=None, operators=None):
        from .dsl import parse_dsl
        return parse_dsl(source, inputs=inputs, parameters=parameters, operators=operators)

    def add_dsl(self, source, *, inputs, parameters=None, operators=None):
        return self.merge(self.from_dsl(source, inputs=inputs, parameters=parameters, operators=operators))

    def register_context(self, context_id, *, anchor, parameters=None, information_cutoff=None):
        if getattr(self, "_store", None) is not None:
            self._store.register_context(self.graph_id, context_id, {"parameters": dict(parameters or {}), "anchor": str(anchor), "information_cutoff": None if information_cutoff is None else str(information_cutoff)})
        else:
            self._contexts = {**getattr(self, "_contexts", {}), context_id: {"parameters": dict(parameters or {}), "anchor": str(anchor), "information_cutoff": None if information_cutoff is None else str(information_cutoff)}}
            self._revision = getattr(self, "_revision", 0) + 1

    def plan_update(self, contexts, *, through, roots=None, input_loader=None, input_proof=None):
        from .update import UpdatePlan
        return UpdatePlan(self, contexts, through=through, roots=roots,
                          input_loader=input_loader, input_proof=input_proof)
