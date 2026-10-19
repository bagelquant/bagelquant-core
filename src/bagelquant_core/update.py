"""Frozen dependency plans exposing caller-admitted node execution."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from threading import RLock, Lock
from uuid import uuid4
from typing import Mapping

import polars as pl

from .execution import ExecutionRuntime
from .graph_management import parents
from .logical import LogicalGraphSpec
from .resources import NativeAdmission, ResourceLimits, resource_limits
from .resources import active_resource_limits
from .preparation import ExecutionPreparation, preparation_scope
from .finite_work import KernelDispatch
from .store import CoreStore, RevisionConflict
from .incremental import (finite_execution_policies, execute_finite_node, execute_checkpoint_node,
                          _finite_time_index_reservation)
from .hashing import hash_mapping


def _same_coordinates(left, right):
    if not left.times.equals(right.times) or left.is_dynamic != right.is_dynamic:
        return False
    if not left.is_dynamic:
        return left.asset_ids.equals(right.asset_ids)
    return left.grid_lazy().sort("time", "asset_id").collect().equals(
        right.grid_lazy().sort("time", "asset_id").collect())


def _plan_context_materializations(spec, bindings, definition, *, capture_contexts=False):
    """Share exact ordinary context/key planning with explicit proof maintenance."""
    from .graph import Graph
    bound = Graph.from_logical_spec(spec, inputs=bindings, parameter_bindings=definition.get("parameters", {}))
    policies = finite_execution_policies(spec, anchor=definition["anchor"])
    operators = {node.node_id: node for node in spec.nodes if node.node_type == "operator"}
    def plan():
        contexts = {node: hash_mapping({"history_policy": policies.get(node),
            "information_cutoff": definition.get("information_cutoff")}) for node in operators}
        planner = ExecutionRuntime(node_contexts=contexts, _capture_contexts=capture_contexts)
        keys = planner.plan_materialization_keys(bound)
        return contexts, planner, keys
    contexts, planner, keys = plan()
    domains = {node.logical_id: node.domain for node in bound.nodes if node.node_type == "input"}
    domains.update(planner.node_domains)
    unsupported = {node for node in policies if str(domains[node].times.min()) != str(definition["anchor"])
        or any(not domains[parent].times.equals(domains[node].times) for parent in parents(operators[node]))}
    if unsupported:
        policies = {node: policy for node, policy in policies.items() if node not in unsupported}
        contexts, planner, keys = plan()
    return bound, policies, contexts, planner, keys


class UpdatePlan:
    """One frozen global or branch update, with no worker/admission policy."""

    def __init__(self, graph, contexts: Mapping, *, through, roots=None,
                 input_loader=None, input_proof=None, input_window_loader=None, input_scope_proof=None,
                 causal_inputs=(), input_version_loader=None, causal_output_domains=None,
                 input_definition_identities: Mapping[str, Mapping[str, str]] | None = None,
                 native_admission: NativeAdmission | None = None, kernel_dispatch: KernelDispatch | None = None,
                 persist_nodes=()):
        if graph._store is None:
            raise ValueError("persistent updates require an explicit CoreStore")
        self.request_id = uuid4().hex
        self.graph = graph
        self.store = graph._store
        self.state = graph._state()
        self.mode = "global" if roots is None else "branch"
        requested = (set(self.state["spec"].outputs.values()) if roots is None else
            {roots} if isinstance(roots, str) else set(roots))
        if roots is not None:
            roots = requested
        self.through = date.fromisoformat(str(through)).isoformat()
        self._lock = RLock()
        self._in_flight = set()
        self._input_loader, self._input_proof = input_loader, input_proof
        self._input_scope_proof = input_scope_proof
        self._input_window_loader = input_window_loader
        self._native_admission = native_admission
        self._kernel_dispatch = kernel_dispatch
        self._input_proof_repairs = {}
        self._preparation = ExecutionPreparation(min(64, active_resource_limits().cache_mib // 4) * 1024 * 1024)
        self._causal_inputs = frozenset(causal_inputs)
        self._input_version_loader = input_version_loader
        self._causal_replays = {}
        self._causal_locks = {context: Lock() for context in contexts}
        self._causal_output_domains = dict(causal_output_domains or {})
        if not set(self._causal_output_domains) <= set(contexts):
            raise ValueError("causal output Domains must belong to admitted contexts")
        self._loaded_inputs = set()
        self._input_locks = {}
        self._parent_locks = {}
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
        from .causal import causal_nodes
        needed_inputs = {node.input_key for node in self.spec.nodes if node.node_type == "input"}
        if not self._causal_inputs <= needed_inputs:
            raise ValueError("causal inputs must be declared in the selected graph")
        self._causal_nodes = causal_nodes(self.spec, self._causal_inputs)
        if input_definition_identities is not None:
            if set(input_definition_identities) != set(contexts):
                raise ValueError("input definition identities must cover exactly the admitted contexts")
            for definitions in input_definition_identities.values():
                if not isinstance(definitions, Mapping) or set(definitions) != needed_inputs:
                    raise ValueError("input definition identities must cover exactly the selected inputs")
                if any(not isinstance(value, str) or not value for value in definitions.values()):
                    raise ValueError("input definition identities must be non-empty stable strings")
        self._input_definitions = {context: dict(input_definition_identities[context])
            if input_definition_identities is not None else {} for context in contexts}
        self._causal_definitions = {}
        for context in contexts:
            closures = {}
            for node in self.spec.nodes:
                if node.node_type == "input":
                    closures[node.node_id] = ({node.input_key: self._input_definitions[context][node.input_key]}
                        if node.input_key in self._input_definitions[context] else {})
                else:
                    closures[node.node_id] = {key: token for parent in parents(node)
                        for key, token in closures[parent].items()}
            self._causal_definitions[context] = closures
        self._nodes = {node.node_id: node for node in self.spec.nodes if node.node_type == "operator"}
        self._inputs = {}
        self._parameters = {}
        self._bound_nodes = {}
        self.keys = {}
        self.domains = {}
        self.policies = {}
        self.numerical_contexts = {}
        self.completion_digests = {}
        self._completion_records = {}
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
            self._input_locks.update({(context_id, source): Lock() for source in needed})
            self._parameters[context_id] = self.state["contexts"].get(context_id, {}).get("parameters", {})
            definition = self.state["contexts"].get(context_id)
            if definition is None:
                raise ValueError("execution context must be explicitly registered")
            bound, self.policies[context_id], self.numerical_contexts[context_id], planner, self.keys[context_id] = (
                _plan_context_materializations(self.spec, bindings, definition))
            if self._causal_nodes:
                from .causal import CAUSAL_EXECUTION_POLICY
                for node_id in self._causal_nodes:
                    self.numerical_contexts[context_id][node_id] = hash_mapping({
                        "history_policy": CAUSAL_EXECUTION_POLICY,
                        **({"input_definitions": self._causal_definitions[context_id][node_id]}
                            if self._causal_definitions[context_id][node_id] else {}),
                        "information_cutoff": definition.get("information_cutoff")})
                planner = ExecutionRuntime(node_contexts=self.numerical_contexts[context_id])
                self.keys[context_id] = planner.plan_materialization_keys(bound)
            self.domains[context_id] = dict(planner.node_domains)
            if context_id in self._causal_output_domains:
                from .materialization import MaterializationKey
                from .domain import Domain
                own = self._causal_output_domains[context_id]
                if not isinstance(own, Domain) or not own.times.equals(next(iter(bindings.values())).domain.times):
                    raise ValueError("causal output Domain must preserve the admitted observation calendar")
                source_assets = set().union(*(set(value.domain.asset_ids.to_list()) for value in bindings.values()))
                if not set(own.asset_ids.to_list()) <= source_assets:
                    raise ValueError("causal output Domain contains undeclared source assets")
                identities = {}
                for node in self.spec.nodes:
                    if node.node_type == "input":
                        continue
                    prior = self.keys[context_id][node.node_id]
                    upstream = tuple(identities.get(token, token) for token in prior.inputs)
                    domain = own if node.node_id in self._causal_nodes else self.domains[context_id][node.node_id]
                    replacement = MaterializationKey(prior.node_id, prior.implementation_id, upstream,
                        domain.signature, prior.context_identity)
                    identities[prior.identity] = replacement.identity
                    self.keys[context_id][node.node_id] = replacement
                    self.domains[context_id][node.node_id] = domain
            self._bound_nodes[context_id] = {node.logical_id: node for node in bound.nodes}
            for node_id, policy in self.policies[context_id].items():
                self._preparation.register_halo(self.domains[context_id][node_id], policy["history_observations"])
            from .incremental import _relevant_parameters
            self.completion_digests[context_id] = {node_id: {
                "definition_digest": hash_mapping({"node": node.to_dict(),
                    "implementation": self.keys[context_id][node_id].implementation_id,
                    "parameters": _relevant_parameters(node, self._parameters[context_id]),
                    **({"input_definitions": self._causal_definitions[context_id][node_id]}
                        if self._causal_definitions[context_id][node_id] else {}),
                    "history_policy": self.policies[context_id].get(node_id)}),
                "upstream_digest": hash_mapping({"inputs": self.keys[context_id][node_id].inputs,
                    "domain": self.keys[context_id][node_id].domain_identity,
                    "context": self.keys[context_id][node_id].context_identity}),
            } for node_id, node in self._nodes.items()}
            self._parent_locks.update({(context_id, node_id): Lock() for node_id in self._nodes})
            completed = self.store.node_completions(self.graph.graph_id, context_id, node_ids=tuple(self._nodes))
            self._completion_records[context_id] = completed
            for node_id, record in completed.items():
                digests = self.completion_digests[context_id][node_id]
                if (record["identity"] == self.keys[context_id][node_id].identity
                        and record["through"] == self.through
                        and all(record[name] == token for name, token in digests.items())):
                    self.results[context_id][node_id] = record["identity"]
                    self.resource_usage[context_id][node_id] = {"outcome": "up_to_date", "computed_blocks": 0,
                        "reused_blocks": 0, "input_sessions": 0, "output_sessions": 0}
        from .regions import plan_regions
        regions = plan_regions(self.spec, policies=self.policies, requested=requested,
            persist_nodes=persist_nodes, causal_nodes=self._causal_nodes)
        # A completed all-durable graph remains reusable when no historical
        # region certificate was authored. Changing physical policy never
        # manufactures evidence or replays that completed history.
        admitted = []
        for region in regions:
            if all(region.output_node not in self._completion_records[context] or any(
                    set(region.nodes) <= set(seal["nodes"]) and (region.output_node not in self.results[context] or
                        all(seal["nodes"].get(node) == self.keys[context][node].to_dict() for node in region.nodes))
                    for seal in self.store.region_evidence(self._completion_records[context][region.output_node]["identity"], kind="seal"))
                    for context in self.results):
                admitted.append(region)
        self.regions = tuple(admitted)
        self._region_outputs = {region.output_node: region for region in self.regions}
        self._transient_owners = {node: region.output_node for region in self.regions
            for node in region.nodes if node != region.output_node}
        self._durable_nodes = set(self._nodes) - self._transient_owners.keys()
        for context in self.results:
            for node in self._transient_owners:
                self.results[context].pop(node, None)
        self.guard()

    @property
    def logical_coverage(self) -> dict:
        """Return satisfied transient nodes and their real durable boundaries."""
        with self._lock:
            return {context: {node: owner for node, owner in self._transient_owners.items() if owner in values}
                for context, values in self.results.items()}

    def _causal_replay(self, context_id):
        from .causal import CausalReplay
        if context_id not in self._causal_replays:
            self._causal_replays[context_id] = CausalReplay(self, context_id)
        return self._causal_replays[context_id]

    def input_proof_windows(self, context_id: str) -> pl.DataFrame:
        """Describe exact potential external proof scopes without consuming values.

        ``required`` scopes belong to the current execution contract;
        ``candidate`` scopes belong to retained metadata-eligible prefixes.
        Candidate proof/state validity is deferred to execution. ``empty``
        consumed prefixes need no scalar token; ``unknown`` strategies require
        an explicit contract before completed incremental work is possible.
        This performs no token callback, value load, audit or index maintenance.
        """
        with preparation_scope(self._preparation):
            return self._input_proof_windows(context_id)

    @property
    def preparation_usage(self) -> dict:
        """Report operation-local immutable reuse; never numerical authority."""
        return self._preparation.report()

    def _input_proof_windows(self, context_id):
        from .incremental import (CANONICAL_SESSIONS, _canonical_halos, _finite_partials,
            _checkpoint_candidates, _checkpoint_parent_scopes)
        from .operator import OPERATOR_REGISTRY
        self.guard()
        if context_id not in self._inputs:
            raise KeyError(f"unknown execution context: {context_id}")
        declared = {node.node_id: node for node in self.spec.nodes}
        rows = {}
        def add(node_id, parent, start, end, cutoff, strategy, status):
            if declared[parent].node_type != "input":
                return
            row = (context_id, declared[parent].input_key, node_id, start, end, cutoff, strategy)
            if row not in rows or status == "required":
                rows[row] = status
        if self._causal_nodes - self.results[context_id].keys():
            for node_id, source, start, end, cutoff in self._causal_replay(context_id).proof_windows():
                if node_id not in self.results[context_id]:
                    parent = next(item.node_id for item in self.spec.nodes if item.node_type == "input" and item.input_key == source)
                    add(node_id, parent, start, end, cutoff, "causal_versions", "required")
        for node_id, node in self._nodes.items():
            if node_id in self.results[context_id] or self._transient_owners.get(node_id) in self.results[context_id]:
                continue
            if node_id in self._causal_nodes:
                continue
            key = self.keys[context_id][node_id]
            if self.store.describe(key.identity) is not None:
                continue  # Exact materialization hits invoke no input proofs.
            domain = self.domains[context_id][node_id]
            direct = {parent: (self._inputs[context_id][declared[parent].input_key]
                if declared[parent].node_type == "input" else self._bound_nodes[context_id][parent])
                for parent in parents(node)}
            if context_id in self._input_proof_repairs:
                from .incremental import _numerical_read_start
                repair = self._input_proof_repairs[context_id]["repair"]
                first, last = date.fromisoformat(repair["from"]), date.fromisoformat(self.through)
                for parent, value in direct.items():
                    add(node_id, parent, _numerical_read_start(value.domain, first, last, repair["history"][node_id]),
                        last, None, "cutoff_tail_repair", "required")
            elif node_id in self.policies[context_id]:
                policy = self.policies[context_id][node_id]
                calendar = domain.times.to_list()
                halos = _canonical_halos(domain, policy["history_observations"])
                partials = _finite_partials(self.store, node_id, key.implementation_id, policy)
                for begin in range(0, len(calendar), CANONICAL_SESSIONS):
                    warm, last = halos[begin], calendar[min(begin + CANONICAL_SESSIONS, len(calendar)) - 1]
                    for parent in direct:
                        add(node_id, parent, warm, last, None, "finite", "required")
                    for candidate in partials.get(str(calendar[begin]), ()):
                        end = date.fromisoformat(candidate["through"])
                        if end <= last and end in calendar[begin:begin + CANONICAL_SESSIONS]:
                            for parent in direct:
                                add(node_id, parent, warm, end, None, "finite", "candidate")
            elif self._checkpoint_supported(node_id):
                operator = node.operator
                if operator is None:
                    raise ValueError("a checkpoint operation must declare its operator")
                registered = OPERATOR_REGISTRY.get(operator)
                if registered.contract.checkpoint_periods is not None:
                    for day in domain.times.to_list():
                        for parent in direct:
                            add(node_id, parent, day, day, None, "checkpoint_periods", "required")
                    continue
                config = dict(self._bound_nodes[context_id][node_id].config())
                config.pop("operator")
                def scopes(end):
                    return _checkpoint_parent_scopes(node, registered.contract, config, domain, direct, end,
                        scoped=self._input_scope_proof is not None)
                def add_scopes(ends, cutoffs, status):
                    for parent in direct:
                        end = None if ends[parent] is None else date.fromisoformat(ends[parent])
                        cutoff = None if parent not in cutoffs else date.fromisoformat(cutoffs[parent])
                        add(node_id, parent, domain.times.min(), end, cutoff, "checkpoint",
                            "empty" if end is None else status)
                add_scopes(*scopes(domain.times.max()), "required")
                calendar_role = registered.contract.checkpoint_calendar_role
                anchor = (() if calendar_role is None else
                    tuple(direct[node.panel_parameters[calendar_role][0]].domain.times.to_list()))
                candidates = _checkpoint_candidates(self.store, node, key, domain, self._parameters[context_id], scopes,
                    calendar_role=calendar_role, anchor_calendar=anchor)
                for candidate in candidates:
                    saved = candidate["execution_policy"]
                    # Legacy absent fields preserve the actual original full
                    # prefix and unscoped cutoff, not today's narrower contract.
                    ends = saved.get("parent_ends", {parent: candidate["through"] for parent in direct})
                    add_scopes(ends, saved.get("parent_cutoffs") or {}, "candidate")
            else:
                for parent in direct:
                    add(node_id, parent, None, None, None, "generic", "unknown")
        self.guard()
        schema = {"context_id": pl.String, "input_key": pl.String, "consumer": pl.String,
            "start": pl.Date, "end": pl.Date, "consumption_cutoff": pl.Date,
            "strategy": pl.String, "status": pl.String}
        return pl.DataFrame([(*row, status) for row, status in rows.items()], schema=schema, orient="row").sort(
            "consumer", "input_key", "start", "end", "consumption_cutoff", "status")

    def guard(self):
        self.store.assert_revision(self.graph.graph_id, self.state["revision"], self.state["state_revision"])

    def ready(self, context_id):
        """Return a stable frontier; caller chooses how many nodes to admit."""
        self.guard()
        with self._lock:
            complete = self.results[context_id]
            return tuple(node_id for node_id, node in self._nodes.items()
                if node_id in self._durable_nodes and node_id not in complete and (context_id, node_id) not in self._in_flight
                and all(parent not in self._nodes or parent in complete for parent in
                    (self._region_outputs[node_id].external_inputs if node_id in self._region_outputs else parents(node))))

    def execute(self, context_id, node_id, *, limits: ResourceLimits | None = None, check_canceled=None):
        """Execute one ready node, reusing already completed immutable parents."""
        from contextlib import nullcontext
        from .operator import OPERATOR_REGISTRY
        from .operation_contract import ExecutionMode

        selected = limits or active_resource_limits()
        reserved_mib = min(64, selected.cache_mib // 4)
        self._preparation.restrict(reserved_mib * 1024 * 1024)
        selected = replace(selected, cache_mib=selected.cache_mib - reserved_mib)
        operation = OPERATOR_REGISTRY.get(self._nodes[node_id].operator)
        columnar = (operation.operation.__module__.startswith("bagelquant_core.operator.")
            and operation.contract.execution == ExecutionMode.LAZY
            and operation.factory is None
            and operation.output_type not in {"prediction", "weights"})
        admission = (nullcontext() if self._native_admission is None else
            self._native_admission("columnar" if columnar else "mixed"))
        with admission, preparation_scope(self._preparation), self.store.publication_context():
            return self._execute(context_id, node_id, limits=selected, check_canceled=check_canceled)

    def _execute(self, context_id, node_id, *, limits, check_canceled):
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
            hit = self.store.describe(key.identity)
            if hit is not None:
                result = self.store.lookup(key, domain=self.domains[context_id][node_id], evidence=False)
                if node_id in self._region_outputs and not any(all(
                        seal["nodes"].get(node) == self.keys[context_id][node].to_dict()
                        for node in self._region_outputs[node_id].nodes)
                        for seal in self.store.region_evidence(key.identity, kind="seal")):
                    raise ValueError("Completed numerical result needs explicit logical-region evidence maintenance")
            elif context_id in self._input_proof_repairs:
                from .cutoff_repair import execute_repair
                self._load_saved_parents(context_id, node_id, guard, selected_parents=(
                    self._region_outputs[node_id].external_inputs if node_id in self._region_outputs else None))
                with resource_limits(replace(limits or ResourceLimits(), parallel_nodes=1)):
                    result, usage = execute_repair(self, context_id, node_id, guard)
                self.resource_usage[context_id].update(usage)
            elif node_id in self._region_outputs:
                from .regions import execute_region
                region = self._region_outputs[node_id]
                self._load_saved_parents(context_id, node_id, guard, selected_parents=region.external_inputs)
                with resource_limits(replace(limits or ResourceLimits(), parallel_nodes=1)):
                    result, usage = execute_region(self, context_id, region, guard)
                self.resource_usage[context_id].update(usage)
            elif node_id in self._causal_nodes:
                with self._causal_locks[context_id], resource_limits(replace(limits or ResourceLimits(), parallel_nodes=1)):
                    replay = self._causal_replay(context_id)
                    result, usage = replay.execute(node_id, guard)
                self.resource_usage[context_id].update(replay.usage)
            elif node_id in self.policies[context_id]:
                self._load_saved_parents(context_id, node_id, guard)
                node_limits = replace(limits or ResourceLimits(), parallel_nodes=1)
                lease = _finite_time_index_reservation(self.domains[context_id][node_id],
                    self.policies[context_id][node_id]["history_observations"],
                    self._nodes[node_id].operator, node_limits)
                with resource_limits(lease.limits), lease:
                    result, usage = execute_finite_node(self.spec, node_id, inputs=self._inputs[context_id],
                        completed=self._values[context_id], parameters=self._parameters[context_id], key=key,
                        domain=self.domains[context_id][node_id], policy=self.policies[context_id][node_id],
                        store=self.store, check_canceled=guard,
                        input_proof=self._proof_reader(context_id),
                        input_window_loader=self._window_reader(context_id, guard, lease.limits,
                            bounded_stored_parents=True),
                        completed_materializations=self.store.completed_materializations(self.graph.graph_id, context_id, node_id),
                        _time_index_lease=lease,
                        _stored_parent_keys={parent: self.keys[context_id][parent]
                            for parent in parents(self._nodes[node_id]) if parent in self.results[context_id]
                            and self.results[context_id][parent] == self.keys[context_id][parent].identity},
                        kernel_dispatch=self._kernel_dispatch,
                        load_inputs=(None if self._input_loader is None or self._input_window_loader is not None else
                            lambda: self._load_inputs(context_id, node_id, guard, lease.limits)))
                self.resource_usage[context_id][node_id] = usage
            elif self._checkpoint_supported(node_id):
                self._load_saved_parents(context_id, node_id, guard)
                with resource_limits(replace(limits or ResourceLimits(), parallel_nodes=1)):
                    result, usage = execute_checkpoint_node(self.spec, node_id, inputs=self._inputs[context_id],
                        completed=self._values[context_id], parameters=self._parameters[context_id], key=key,
                        domain=self.domains[context_id][node_id], store=self.store, check_canceled=guard,
                        input_proof=self._proof_reader(context_id),
                        completed_materializations=self.store.completed_materializations(self.graph.graph_id, context_id, node_id),
                        load_inputs=(None if self._input_loader is None else
                            lambda: self._load_inputs(context_id, node_id, guard, limits)),
                        input_window_loader=self._window_reader(context_id, guard, limits),
                        input_scope_proof=self._proof_reader(context_id) if self._input_scope_proof is not None else None)
                self.resource_usage[context_id][node_id] = usage
            else:
                from .incremental import _relevant_parameters
                bindings = _relevant_parameters(self._nodes[node_id], self._parameters[context_id])
                for identity in self.store.completed_materializations(self.graph.graph_id, context_id, node_id):
                    saved = self.store.describe(identity)
                    if saved is None:
                        raise ValueError("Completed numerical authority is missing; explicit Core recovery is required")
                    if saved["key"]["implementation_id"] != key.implementation_id:
                        continue
                    prior_bindings = (saved["execution_policy"] or {}).get("parameter_bindings")
                    if not isinstance(prior_bindings, Mapping) or prior_bindings == bindings:
                        raise ValueError("Completed numerical history has no declared finite or checkpoint contract; "
                            "plan affected intervals or run explicit numerical-policy/input-proof maintenance before updating")
                self._load_inputs(context_id, node_id, guard, limits)
                self._load_saved_parents(context_id, node_id, guard)
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
            self.store.complete_node(self.graph.graph_id, context_id, key,
                revision=self.state["revision"], state_revision=self.state["state_revision"],
                through=self.through,
                checked_through=self.state["contexts"][context_id].get("information_cutoff") or self.through,
                **self.completion_digests[context_id][node_id])
            with self._lock:
                self.results[context_id][node_id] = result.key.identity
                self._values[context_id][node_id] = result.panel
            return result
        finally:
            with self._lock:
                self._in_flight.discard((context_id, node_id))

    def _window_reader(self, context_id, guard, limits, *, bounded_stored_parents=False):
        loader = self._input_window_loader
        if loader is None and not bounded_stored_parents:
            return None
        sources = {node.node_id: node.input_key for node in self.spec.nodes if node.node_type == "input"}
        def read(parent, expected, start, end):
            if parent not in sources:
                if not bounded_stored_parents or not isinstance(self.store, CoreStore):
                    return expected
                with self._lock:
                    key = self.keys[context_id].get(parent)
                    identity = self.results[context_id].get(parent)
                if key is None or identity is None:
                    return expected
                if identity != key.identity:
                    raise ValueError("completed parent numerical authority differs from the frozen key")
                if expected.identity != identity:
                    return expected
                if expected.domain.signature != key.domain_identity:
                    raise ValueError("completed parent Domain differs from the frozen key")
                guard()
                frame = self.store.read_values(identity, start=start, end=end, include_traces=True)
                from .node import Node
                value = Node.from_domain(frame, expected.domain.window(start, end),
                    name=expected.name, metadata=expected.metadata, source_key=expected.source_key,
                    identity=expected.identity, value_type=expected.value_type,
                    trace_columns=expected.trace_columns, trace_identity=expected.trace_identity)
                guard()
                return value
            if loader is None:
                return expected
            guard()
            with resource_limits(limits or ResourceLimits()):
                value = loader(context_id, sources[parent], start, end)
            own = expected.domain.window(start, end)
            coordinates_equal = _same_coordinates(value.domain, own)
            if (value.identity != expected.identity or value.domain.signature != own.signature or not coordinates_equal
                    or value.value_type != expected.value_type or value.trace_columns != expected.trace_columns
                    or value.trace_identity != expected.trace_identity):
                raise ValueError("bounded input differs from the frozen source/window authority")
            guard()
            return value
        return read

    def _load_saved_parents(self, context_id, node_id, guard, *, selected_parents=None):
        """Construct lazy typed plans only for completed parents now consumed."""
        for parent in parents(self._nodes[node_id]) if selected_parents is None else selected_parents:
            if parent not in self._nodes:
                continue
            with self._parent_locks[(context_id, parent)]:
                guard()
                with self._lock:
                    if parent in self._values[context_id]:
                        continue
                    identity = self.results[context_id].get(parent)
                if identity is None:
                    raise ValueError("a computed parent is incomplete")
                key = self.keys[context_id][parent]
                value = self.store.lookup(key, domain=self.domains[context_id][parent], evidence=False)
                if value is None or identity != value.key.identity:
                    raise ValueError("completed parent numerical authority is missing; explicit Core recovery is required")
                guard()
                with self._lock:
                    self._values[context_id][parent] = value.panel

    def _load_inputs(self, context_id, node_id, guard, limits):
        """Coalesce direct source reads; completed parents already own values."""
        if self._input_loader is None:
            return
        sources = {node.node_id: node.input_key for node in self.spec.nodes if node.node_type == "input"}
        for parent in parents(self._nodes[node_id]):
            if parent not in sources:
                continue
            source = sources[parent]
            locator = (context_id, source)
            with self._input_locks[locator]:
                guard()
                if locator in self._loaded_inputs:
                    continue
                expected = self._inputs[context_id][source]
                with resource_limits(limits or ResourceLimits()):
                    value = self._input_loader(context_id, source)
                coordinates_equal = _same_coordinates(value.domain, expected.domain)
                if (value.identity != expected.identity or value.domain.signature != expected.domain.signature or not coordinates_equal
                        or value.value_type != expected.value_type or value.trace_columns != expected.trace_columns
                        or value.trace_identity != expected.trace_identity):
                    raise ValueError("immutable source identity changed after planning")
                guard()
                self._inputs[context_id][source] = value
                self._loaded_inputs.add(locator)

    def _proof_reader(self, context_id):
        def proof(parent, value, start, end, consumption_cutoff=None):
            locator = (context_id, parent, value.identity, value.domain, value.value_type,
                value.trace_columns, value.trace_identity, start, end, consumption_cutoff)
            return self._preparation.remember("interval_proof", locator,
                lambda: read(parent, value, start, end, consumption_cutoff), lambda token: len(token) * 4)

        def read(parent, value, start, end, consumption_cutoff):
            token = self.store.interval_identity(value.identity, start=start, end=end)
            if token is not None:
                return token
            node = next(item for item in self.spec.nodes if item.node_id == parent)
            if node.node_type == "input":
                if consumption_cutoff is not None and self._input_scope_proof is None:
                    raise ValueError("a causal consumption cutoff requires an explicit scoped input proof")
                reader = self._input_proof if consumption_cutoff is None else self._input_scope_proof
                if reader is not None:
                    token = (reader(context_id, node.input_key, start, end) if consumption_cutoff is None else
                        reader(context_id, node.input_key, start, end, consumption_cutoff))
                    if token is None:
                        return None
                    # A caller's explicit Domain identity is opaque evidence;
                    # distinguish actual immutable Domain objects in this
                    # operation-local memo, retaining them against id reuse.
                    return value.source_interval_identity(token, start, end)
            return value.interval_identity(start, end)
        return proof

    def _checkpoint_supported(self, node_id):
        from .operator import OPERATOR_REGISTRY
        operator = self._nodes[node_id].operator
        if operator is None:
            raise ValueError("a checkpoint operation must declare its operator")
        return OPERATOR_REGISTRY.get(operator).contract.checkpoint_replay

    @property
    def complete(self):
        with self._lock:
            return not self._in_flight and all(set(values) == self._durable_nodes for values in self.results.values())

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
            receipt = self.store.commit_update(self.graph.graph_id, revision=self.state["revision"],
                state_revision=self.state["state_revision"], expected_previous=self.state["current_update"],
                results=self.results, through=self.through, mode=self.mode, request_id=self.request_id,
                coverage=self.logical_coverage)
            self._preparation.clear()
            return receipt
