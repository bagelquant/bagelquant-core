"""Lazy sparse graph execution with explicit dense and eager barriers."""

from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from contextvars import copy_context
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
import polars as pl

from bagelquant_core.frame import ASSET_ID, TIME, VALUE
from bagelquant_core.graph import Graph
from bagelquant_core.hashing import hash_mapping
from bagelquant_core.node import Node
from bagelquant_core.materialization import MaterializationKey, MaterializationStore, NodeMaterialization, MaterializationStatus, materialization_trace_identity
from bagelquant_core.operator_state import (
    capture_node_evidence, capture_operator_checkpoints, current_operator_checkpoints,
    operator_input_context, replay_node_evidence,
)
from bagelquant_core.operation_contract import (
    ExecutionMode,
    InputDensity,
    OperationContract,
    TraceRule,
)
from bagelquant_core.node import Domain
from bagelquant_core.resources import active_resource_limits, resource_limits

logger = logging.getLogger(__name__)

_TIME_ASSET_ORDER = "time_asset"
_ASSET_TIME_ORDER = "asset_time"
EXECUTION_KERNEL_VERSION = "logical_runtime.v2"


@dataclass(frozen=True, slots=True)
class PlanValue:
    frame: pl.LazyFrame
    domain: Domain
    density: InputDensity
    identity: str
    trace_columns: tuple[str, ...] = ()
    trace_frame: pl.LazyFrame | None = None
    trace_identity: str | None = None
    default_value: float | None = None
    categorical: bool = False
    prediction: bool = False
    cacheable: bool = True
    key_identity: str | None = None
    order: str | None = None
    trace_key_identity: str | None = None
    trace_order: str | None = None
    exact_domain: bool = False
    asset_time_ordered: bool = False
    validated_keys: bool = False
    physical_identity: str | None = None


@dataclass(frozen=True, slots=True)
class _EagerLayout:
    asset_permutation: np.ndarray
    inverse_permutation: np.ndarray
    asset_offsets: np.ndarray
    time_offsets: np.ndarray


@dataclass(frozen=True, slots=True)
class _EagerWork:
    node: Node
    prepared: tuple[PlanValue, ...]
    domain: Domain
    physical_identity: str


class _DeferredEager(Exception):
    def __init__(self, work: _EagerWork) -> None:
        self.work = work


class ExecutionRuntime:
    """Compile graphs into Polars plans and materialize only public outputs."""

    def __init__(self, alignment: str = "inner", *,
                 materialization_store: MaterializationStore | None = None,
                  context_identity: str = "",
                  node_contexts: Mapping[str, str] | None = None, check_canceled=None) -> None:
        if alignment != "inner":
            raise ValueError("ExecutionRuntime only supports inner alignment")
        self.check_canceled = check_canceled or (lambda: None)
        self.cache: dict[str, Node] = {}
        self._plan_cache: dict[str, PlanValue] = {}
        self._physical_plan_cache: dict[str, PlanValue] = {}
        self._alignment = alignment
        self.materialization_store = materialization_store
        self.context_identity = context_identity
        self.node_contexts = dict(node_contexts or {})
        if any(not isinstance(node_id, str) or not node_id or not isinstance(context, str)
               for node_id, context in self.node_contexts.items()):
            raise ValueError("node execution contexts require nonempty logical IDs and string contexts")
        self.node_materializations: dict[str, NodeMaterialization] = {}
        self.node_domains: dict[str, Domain] = {}
        self.node_artifacts: dict[str, Mapping[str, Any]] = {}
        self.node_checkpoints = {}
        self.node_training_audits = {}
        self._pending_materializations: dict[str, tuple[Node, PlanValue, MaterializationKey]] = {}
        self._loaded_materializations: dict[str, NodeMaterialization] = {}
        self._run_evidence = None
        self._active_evaluated: dict[int, PlanValue] | None = None
        self._pending_estimated_bytes = 0
        self.persistent_hits = 0
        self.persistent_misses = 0
        self.persistent_partials = 0
        self._active_eager_inputs: dict[str, pl.DataFrame] | None = None
        self._active_eager_results: dict[
            tuple[object, ...],
            tuple[pl.DataFrame, pl.DataFrame],
        ] | None = None
        self._active_eager_layouts: dict[
            tuple[str, str, str],
            _EagerLayout,
        ] | None = None
        self._active_eager_physical_results: dict[
            str,
            pl.DataFrame,
        ] | None = None
        self.materializations = 0
        self.eager_barriers = 0
        self._defer_eager = False
        self._completed_eager: dict[int, pl.DataFrame] = {}
        self._effective_limits = active_resource_limits()
        self._last_memory_sample = 0.0
        self.resource_usage = {"max_parallel_nodes": 0, "eager_batches": 0}
        self._diagnostics = {
            "semantic_cse_hits": 0,
            "alignments_elided": 0,
            "sorts_elided": 0,
            "membership_applications": 0,
            "positional_operator_hits": 0,
            "eager_cse_hits": 0,
            "solver_batches": 0,
            "active_window_iterations": 0,
        }

    def _remember_output(self, identity: str, panel: Node) -> None:
        """Bound retained numerical output caches without changing graph identity."""
        limit = self._effective_limits.cache_mib * 1024 * 1024
        def size(value):
            frame = value._cached_dense
            return 0 if frame is None else frame.estimated_size()
        required = size(panel)
        if required > limit or limit == 0:
            self.cache.clear()
            return
        self.cache.pop(identity, None)
        resident = sum(size(value) for value in self.cache.values())
        while self.cache and resident + required > limit:
            resident -= size(self.cache.pop(next(iter(self.cache))))
        self.cache[identity] = panel

    def plan_materialization_keys(self, graph: Graph) -> dict[str, MaterializationKey]:
        """Plan operator identities without collecting or evaluating numerical inputs.

        Uses the same full-calendar context as durable execution. Callers may
        bind scan-backed or schema-correct placeholder inputs with proven
        identities, query complete results, then load only missing payloads.
        """
        if isinstance(graph, Node):
            graph = graph.graph
        if not isinstance(graph, Graph):
            raise TypeError("materialization planning expects a Graph")
        values: dict[int, PlanValue] = {}
        keys: dict[str, MaterializationKey] = {}
        self.node_domains = {}
        with ExitStack() as scope:
            if current_operator_checkpoints() is None:
                calendar = next((node.domain.times.to_list() for node in graph.nodes if node.node_type == "input"), ())
                scope.enter_context(capture_operator_checkpoints(calendar=calendar))
            for node in graph.topological_sort():
                if node.node_type == "input":
                    value = self._panel_plan(node)
                    value = replace(value, physical_identity=getattr(node, "_materialization_identity", None) or hash_mapping({"input": node.identity,
                        "trace": node.trace_identity, "trace_columns": node.trace_columns,
                        "domain": node.domain.signature, "categorical": (isinstance(node, Node) and node.value_type == "category"),
                        "prediction": (isinstance(node, Node) and node.value_type == "prediction")}))
                else:
                    parents = tuple(values[id(parent)] for parent in node.parents)
                    projected = node.config().get("operator", "").endswith(".project_domain")
                    domain = parents[-1].domain if projected else self._resolve_domain(parents)
                    key = self._materialization_key(node, parents, domain)
                    keys[node.logical_id] = key
                    self.node_domains[node.logical_id] = domain
                    value = PlanValue(frame=parents[0].frame, domain=domain,
                        density=self._contract(node).density, identity=key.identity,
                        trace_columns=tuple(dict.fromkeys(trace for parent in parents for trace in parent.trace_columns)),
                        categorical=node.value_type == "category",
                        prediction=node.value_type == "prediction",
                        physical_identity=key.identity)
                values[id(node)] = value
        return keys

    def _run_frontiers(self, graph: Graph, evaluated: dict[int, PlanValue]) -> None:
        """Keep lazy work fused and admit bounded independent eager kernels."""
        nodes = graph.topological_sort()
        remaining = {id(node): len({id(parent) for parent in node.parents}) for node in nodes}
        children: dict[int, list[Node]] = {}
        for node in nodes:
            for parent_id in dict.fromkeys(id(parent) for parent in node.parents):
                children.setdefault(parent_id, []).append(node)
        ready = deque(node for node in nodes if remaining[id(node)] == 0)

        def completed(node: Node) -> None:
            for child in children.get(id(node), ()):
                remaining[id(child)] -= 1
                if remaining[id(child)] == 0:
                    ready.append(child)

        with ThreadPoolExecutor(max_workers=self._effective_limits.parallel_nodes,
                                thread_name_prefix="bagelquant-kernel") as executor:
            while ready:
                frontier: list[_EagerWork] = []
                self._defer_eager = True
                try:
                    while ready:
                        node = ready.popleft()
                        try:
                            self._run_node(node, evaluated)
                        except _DeferredEager as deferred:
                            frontier.append(deferred.work)
                        else:
                            completed(node)
                finally:
                    self._defer_eager = False
                groups: dict[object, list[_EagerWork]] = {}
                for work in frontier:
                    groups.setdefault(self._eager_group_key(work), []).append(work)
                waiting = deque(groups.values())
                while waiting:
                    limits = self._effective_limits
                    headroom = max(0, limits.memory_target_mib) * 1024 * 1024
                    batch: list[list[_EagerWork]] = []
                    estimate = 0
                    while waiting and len(batch) < limits.parallel_nodes:
                        group = waiting[0]
                        work = group[0]
                        required = self._estimate_plan_bytes(work.domain, 0) * (len(work.prepared) + 2)
                        if batch and estimate + required > headroom:
                            break
                        batch.append(waiting.popleft())
                        estimate += required
                    self.resource_usage["max_parallel_nodes"] = max(
                        self.resource_usage["max_parallel_nodes"], len(batch))
                    self.resource_usage["eager_batches"] += 1
                    # Planning a later ready branch may have spilled an earlier
                    # branch. Do not collect the earlier captured lazy plan again.
                    for index, group in enumerate(batch):
                        refreshed = []
                        for work in group:
                            parents = tuple(self._frozen_output_plan(None, parent) for parent in work.prepared)
                            prepared = tuple(self._ensure_dense(parent) if self._contract(work.node).density
                                == InputDensity.DENSE_REQUIRED else self._expand_implicit(parent) for parent in parents)
                            refreshed.append(replace(work, prepared=prepared))
                        batch[index] = refreshed
                    prepared = tuple(value for group in batch for value in group[0].prepared)
                    frames = self._collect_eager_inputs(prepared,
                        tuple(value.frame.select(TIME, ASSET_ID, VALUE) for value in prepared))
                    offset = 0
                    jobs = []
                    for group in batch:
                        count = len(group[0].prepared)
                        inputs = frames[offset:offset + count]
                        offset += count
                        context = copy_context()
                        arguments = (group, inputs, limits.for_workers(len(batch)))
                        jobs.append(executor.submit(context.run, self._execute_eager_group, *arguments))
                    # Merge evidence and publish in stable graph order, not completion order.
                    for group, job in zip(batch, jobs, strict=True):
                        outputs, evidence, diagnostics = job.result()
                        for name, count in diagnostics.items():
                            self._diagnostics[name] += count
                        for work, output in zip(group, outputs, strict=True):
                            node_id = work.node.logical_id
                            replay_node_evidence(node_id,
                                artifacts=evidence.artifacts.get(node_id, {}),
                                checkpoint=evidence.checkpoints.get(node_id),
                                training_audits=evidence.training_audits.get(node_id, ()))
                            self._completed_eager[id(work.node)] = output
                            self._run_node(work.node, evaluated)
                            completed(work.node)
                    del frames, jobs

    def _eager_group_key(self, work: _EagerWork) -> object:
        node = work.node
        if self._is_builtin_operation(node) and self._contract(node).deterministic:
            operation = node.operation.display_name
            if operation in {"rolling_rank", "rolling_percentile"}:
                parameters = self._node_parameters(node)
                parent = work.prepared[0]
                return ("rolling_rank_pair", parent.identity, work.domain.signature,
                    parameters.get("window"), parameters.get("min_periods"))
            return work.physical_identity
        return id(node)

    @staticmethod
    def _execute_eager_group(group: list[_EagerWork], inputs: tuple[pl.DataFrame, ...], limits):
        from bagelquant_core.operator.training import capture_training_audits

        local = ExecutionRuntime()
        local._active_eager_inputs = {}
        local._active_eager_results = {}
        local._active_eager_layouts = {}
        checkpoint_context = current_operator_checkpoints()
        restored = None if checkpoint_context is None else checkpoint_context.restored
        calendar = () if checkpoint_context is None else checkpoint_context.calendar
        results = []
        with resource_limits(limits), capture_node_evidence() as evidence, \
                capture_operator_checkpoints(restored, calendar=calendar), capture_training_audits():
            first = group[0]
            result = local._compute_eager(first.node, first.prepared, inputs, first.domain)
            results.append(result)
            for work in group[1:]:
                if work.physical_identity == first.physical_identity:
                    results.append(result)
                else:
                    results.append(local._compute_eager(work.node, work.prepared, inputs, work.domain))
        return results, evidence, local._diagnostics

    def run(
        self,
        graph: Graph,
        *,
        dense_output: bool = True,
    ) -> Node | Mapping[str, Node]:
        if isinstance(graph, Node):
            graph = graph.graph
        if not isinstance(graph, Graph):
            raise TypeError("ExecutionRuntime.run expects a Graph")
        if self.materialization_store is not None and any(node.node_type == "input" and not getattr(node, "_durable_identity", False) for node in graph.nodes):
            raise ValueError("persistent execution requires immutable source evidence for lazy inputs")
        if self._active_eager_inputs is not None:
            raise RuntimeError("ExecutionRuntime does not support nested runs")
        guards = tuple(dict.fromkeys((*getattr(graph, "_execution_guards", ()), *(guard for node in graph.nodes for guard in getattr(node, "_execution_guards", ())))))
        for guard in guards:
            guard()
        previous_check = self.check_canceled
        def check():
            previous_check()
            for guard in guards:
                guard()
        self.check_canceled = check
        self._active_eager_inputs = {}
        self._active_eager_results = {}
        self._active_eager_layouts = {}
        self._active_eager_physical_results = {}
        self.node_materializations = {}
        self.node_domains = {}
        self.node_artifacts = {}
        self._pending_materializations = {}
        self._loaded_materializations = {}
        self._pending_estimated_bytes = 0
        self._effective_limits = active_resource_limits()
        self._completed_eager = {}
        scope = ExitStack()
        self._run_evidence = scope.enter_context(capture_node_evidence())
        if self.materialization_store is not None and current_operator_checkpoints() is None:
            calendar = next((node.domain.times.to_list() for node in graph.nodes if node.node_type == "input"), ())
            scope.enter_context(capture_operator_checkpoints(calendar=calendar))
        try:
            evaluated: dict[int, PlanValue] = {}
            self._active_evaluated = evaluated
            if self._effective_limits.parallel_nodes > 1:
                self._run_frontiers(graph, evaluated)
            plans = [
                (node, self._run_node(node, evaluated))
                for node in graph._outputs
            ]
            if self.materialization_store is not None:
                self._publish_materializations()
                plans = [(node, self._frozen_output_plan(node, plan)) for node, plan in plans]
            if dense_output:
                results = self._materialize_many(plans)
                return graph._present_outputs(results)

            results: dict[str, Node] = {}
            for node, plan in plans:
                output = self._materialize_node(
                    node,
                    plan,
                    dense_output=False,
                )
                node.set_output(output)
                results[node.name] = output
            return graph._present_outputs(results)
        finally:
            self.check_canceled = previous_check
            self.node_artifacts = dict(self._run_evidence.artifacts)
            self.node_checkpoints = dict(self._run_evidence.checkpoints)
            self.node_training_audits = dict(self._run_evidence.training_audits)
            scope.close()
            self._run_evidence = None
            self._active_evaluated = None
            self._active_eager_inputs = None
            self._active_eager_results = None
            self._active_eager_layouts = None
            self._completed_eager.clear()
            self._defer_eager = False
            self._active_eager_physical_results = None

    def _materialize_many(
        self,
        outputs: list[tuple[Node, PlanValue]],
        *, dense_output: bool = True,
    ) -> Mapping[str, Node]:
        """Collect all uncached outputs in one Polars execution boundary."""

        results: dict[str, Node] = {}
        pending: list[tuple[Node, PlanValue, Node]] = []
        for node, plan in outputs:
            plan = self._expand_implicit(plan)
            cached = self.cache.get(plan.identity) if plan.cacheable and dense_output else None
            if cached is not None:
                node.set_output(cached)
                results[node.name] = cached
                continue
            panel_type = Node
            panel = panel_type._from_plan(
                self._frame_with_traces(plan),
                value_type=node.value_type,
                domain=plan.domain,
                name=node.name,
                metadata=node.metadata,
                identity=plan.identity,
                trace_identity=plan.trace_identity,
                trace_columns=plan.trace_columns,
                dense_output=False,
                validated_keys=plan.validated_keys,
                exact_domain=plan.exact_domain,
                key_identity=plan.key_identity,
            )
            pending.append((node, plan, panel))

        if pending:
            value_plans: list[pl.LazyFrame] = []
            key_owner_by_domain: dict[str, int] = {}
            key_owner_indices: list[int] = []
            for index, (_, plan, _) in enumerate(pending):
                dense_plan = self._dense_output_plan(plan, plan.frame) if dense_output else plan.frame
                owner_index = key_owner_by_domain.setdefault(plan.domain.signature, index) if dense_output else index
                key_owner_indices.append(owner_index)
                value_plans.append(
                    dense_plan
                    if owner_index == index
                    else dense_plan.select(VALUE)
                )
            trace_plans: dict[
                tuple[str, str, str | None, tuple[str, ...]],
                pl.LazyFrame,
            ] = {}
            trace_keys: list[
                tuple[str, str, str | None, tuple[str, ...]] | None
            ] = []
            for _, plan, _ in pending:
                if (
                    plan.trace_frame is None
                    or plan.trace_identity is None
                ):
                    trace_keys.append(None)
                    continue
                key = (
                    plan.domain.signature,
                    plan.trace_identity,
                    plan.trace_key_identity,
                    plan.trace_columns,
                )
                trace_keys.append(key)
                trace_frame = plan.trace_frame.select(TIME, ASSET_ID, *plan.trace_columns)
                trace_plans.setdefault(
                    key,
                    self._dense_output_plan(plan, trace_frame, order=plan.trace_order)
                    if dense_output else trace_frame,
                )
            unique_trace_keys = list(trace_plans)
            collected = pl.collect_all(
                [
                    *value_plans,
                    *(trace_plans[key] for key in unique_trace_keys),
                ]
            )
            self.materializations += 1
            trace_frames = {
                key: frame
                for key, frame in zip(
                    unique_trace_keys,
                    collected[len(pending) :],
                    strict=True,
                )
            }
            for index, (node, plan, panel) in enumerate(pending):
                owner_index = key_owner_indices[index]
                frame = (
                    collected[index]
                    if owner_index == index
                    else collected[owner_index]
                    .select(TIME, ASSET_ID)
                    .with_columns(collected[index].get_column(VALUE))
                )
                trace_key = trace_keys[index]
                if trace_key is not None:
                    frame = frame.hstack(
                        trace_frames[trace_key].select(
                            *plan.trace_columns
                        )
                    )
                if plan.validated_keys:
                    if (
                        not plan.categorical
                        and frame.schema[VALUE].is_float()
                    ):
                        frame = frame.with_columns(
                            pl.col(VALUE).fill_nan(None)
                        )
                    validated = frame.select(
                        TIME,
                        ASSET_ID,
                        VALUE,
                        *plan.trace_columns,
                    )
                else:
                    validated = panel._validate_collected(frame)
                if dense_output:
                    panel._cached_dense = validated
                else:
                    panel._frame = validated.lazy()
                    panel._validated_keys = True
                    panel._exact_domain = panel.domain._contains_exact_keys(validated)
                    if panel._exact_domain:
                        panel._cached_dense = validated
                if plan.cacheable and dense_output:
                    self._remember_output(plan.identity, panel)
                node.set_output(panel)
                results[node.name] = panel
        return {
            node.name: results[node.name]
            for node, _ in outputs
        }

    @staticmethod
    def _panel_plan(panel: Node) -> PlanValue:
        return PlanValue(
            frame=panel.lazy(include_traces=False), domain=panel.domain,
            density=InputDensity.SPARSE_OK, identity=panel.identity,
            trace_columns=panel.trace_columns,
            trace_frame=panel.lazy(include_traces=True).select(TIME, ASSET_ID, *panel.trace_columns) if panel.trace_columns else None,
            trace_identity=panel.trace_identity, categorical=(isinstance(panel, Node) and panel.value_type == "category"),
            prediction=(isinstance(panel, Node) and panel.value_type == "prediction"), key_identity=panel._key_identity,
            order=_TIME_ASSET_ORDER, trace_key_identity=panel._key_identity,
            trace_order=_TIME_ASSET_ORDER, exact_domain=panel._exact_domain,
            asset_time_ordered=True, validated_keys=panel._validated_keys,
            physical_identity=panel.identity,
        )

    def _materialization_key(self, node: Node, parents: tuple[PlanValue, ...], domain: Domain) -> MaterializationKey:
        from bagelquant_core.operator import OPERATOR_REGISTRY

        operator = OPERATOR_REGISTRY.get(node.config()["operator"])
        implementation = hash_mapping({"operator": operator.registry_name, "version": operator.version,
            "runtime_kernel": EXECUTION_KERNEL_VERSION,
            "execution": operator.contract.execution, "density": operator.contract.density,
            "trace_rule": operator.contract.trace_rule, "output_type": operator.output_type})
        context = hash_mapping({"context": self.node_contexts.get(node.logical_id, self.context_identity),
                                "operator": operator_input_context(node.logical_id),
                                "parameters": self._node_parameters(node)})
        return MaterializationKey(node.logical_id, implementation,
            tuple(parent.physical_identity or parent.identity for parent in parents), domain.signature, context)

    def _record_materialization_plan(self, node: Node, value: PlanValue, key: MaterializationKey) -> PlanValue:
        if self.materialization_store is not None and value.cacheable:
            if key.identity not in self._pending_materializations:
                self._pending_materializations[key.identity] = (node, value, key)
                self._pending_estimated_bytes += self._estimate_plan_bytes(value.domain, len(value.trace_columns))
        return replace(value, physical_identity=key.identity)

    @staticmethod
    def _estimate_plan_bytes(domain: Domain, traces: int) -> int:
        # Conservative key/value/string offset plus trace-date estimate. Sparse
        # plans are never expanded merely to measure admission requirements.
        return domain.size * (40 + traces * 8)

    def _admit_materialization(self, domain: Domain, parents: tuple[PlanValue, ...]) -> tuple[PlanValue, ...]:

        # A later sibling's recursive execution may have published an earlier
        # parent after its PlanValue was captured. Refresh that local reference
        # even when this node does not itself cross an admission boundary.
        parents = tuple(self._frozen_output_plan(None, parent) for parent in parents)
        if self.materialization_store is None or not self._pending_materializations:
            return parents
        limits = self._effective_limits
        available = min(limits.cache_mib, max(1, limits.memory_target_mib // 4)) * 1024 * 1024
        estimate = self._estimate_plan_bytes(domain, len({trace for parent in parents for trace in parent.trace_columns}))
        if self._pending_estimated_bytes + estimate > available:
            self._publish_materializations()
            return tuple(self._frozen_output_plan(None, parent) for parent in parents)
        return parents

    def _publish_materializations(self) -> None:
        """Freeze sparse intermediate plans together at the existing collect boundary."""
        pending = list(self._pending_materializations.values())
        if not pending:
            return
        panels = self._materialize_many([(node, plan) for node, plan, _ in pending], dense_output=False)
        for node, plan, key in pending:
            panel = panels[node.name]
            frozen = Node.from_domain(panel.lazy(include_traces=True), panel.domain, value_type=panel.value_type,
                name=node.logical_id, identity=key.identity, trace_identity=materialization_trace_identity(key, panel.trace_columns),
                trace_columns=panel.trace_columns, source_key=node.logical_id)
            frozen._validated_keys = True
            frozen._exact_domain = panel._exact_domain
            frozen._cached_dense = panel._cached_dense
            record = NodeMaterialization(key, frozen,
                artifacts=dict(self._run_evidence.artifacts.get(node.logical_id, {})),
                checkpoint=self._run_evidence.checkpoints.get(node.logical_id),
                training_audits=tuple(self._run_evidence.training_audits.get(node.logical_id, ())))
            self.check_canceled()
            self.materialization_store.publish(record)
            saved = self.materialization_store.query(key)
            if saved.status != MaterializationStatus.HIT or saved.materialization is None or saved.materialization.key != key:
                raise ValueError("materialization publication did not produce an exact immutable receipt")
            record = saved.materialization
            node.set_output(record.panel)
            self.node_materializations[node.logical_id] = record
            self._loaded_materializations[key.identity] = record
        self._pending_materializations.clear()
        self._pending_estimated_bytes = 0
        if self._active_evaluated is not None:
            for node_id, plan in list(self._active_evaluated.items()):
                self._active_evaluated[node_id] = self._frozen_output_plan(None, plan)
        # Persisted boundaries replace old plans with scan-backed inputs.
        # Retaining the eager CSE arrays would defeat resource admission.
        self._plan_cache.clear()
        self._physical_plan_cache.clear()
        self.cache.clear()
        for cache in (self._active_eager_inputs, self._active_eager_results,
                      self._active_eager_layouts, self._active_eager_physical_results):
            if cache is not None:
                cache.clear()

    def _frozen_output_plan(self, node: Node | None, plan: PlanValue) -> PlanValue:
        record = self._loaded_materializations.get(plan.physical_identity or "")
        if record is None:
            return plan
        value = self._panel_plan(record.panel)
        return replace(value, identity=plan.identity, physical_identity=record.key.identity)

    def _dense_output_plan(
        self,
        plan: PlanValue,
        frame: pl.LazyFrame,
        *,
        order: str | None = None,
    ) -> pl.LazyFrame:
        """Align only subset plans and sort exact plans only when required."""

        resolved_order = plan.order if order is None else order
        if not plan.exact_domain:
            columns = frame.collect_schema().names()
            traces = tuple(
                column
                for column in columns
                if column not in {TIME, ASSET_ID, VALUE}
            )
            if VALUE in columns:
                return plan.domain.align_lazy(
                    frame,
                    trace_columns=traces,
                )
            return (
                plan.domain.grid_lazy()
                .join(
                    frame,
                    on=[TIME, ASSET_ID],
                    how="left",
                    maintain_order="left",
                )
            )
        self._diagnostics["alignments_elided"] += 1
        if resolved_order == _TIME_ASSET_ORDER:
            return frame
        return frame.sort([TIME, ASSET_ID])

    def _run_node(
        self,
        node: Node,
        evaluated: dict[int, PlanValue],
    ) -> PlanValue:
        self.check_canceled()
        node_id = id(node)
        if node_id in evaluated:
            return evaluated[node_id]
        if node.node_type == "input":
            value = PlanValue(
                frame=node._frame.select(TIME, ASSET_ID, VALUE),
                domain=node.domain,
                density=InputDensity.SPARSE_OK,
                identity=node.identity,
                trace_columns=node.trace_columns,
                trace_frame=(
                    node._frame.select(
                        TIME, ASSET_ID, *node.trace_columns
                    )
                    if node.trace_columns
                    else None
                ),
                trace_identity=node.trace_identity,
                categorical=(isinstance(node, Node) and node.value_type == "category"),
                prediction=(isinstance(node, Node) and node.value_type == "prediction"),
                key_identity=node._key_identity,
                order=_TIME_ASSET_ORDER,
                trace_key_identity=node._key_identity,
                trace_order=_TIME_ASSET_ORDER,
                exact_domain=node._exact_domain,
                asset_time_ordered=True,
                validated_keys=node._validated_keys,
                physical_identity=getattr(node, "_materialization_identity", None) or hash_mapping({"input": node.identity, "trace": node.trace_identity,
                    "trace_columns": node.trace_columns, "domain": node.domain.signature,
                    "categorical": (isinstance(node, Node) and node.value_type == "category"), "prediction": (isinstance(node, Node) and node.value_type == "prediction")}),
            )
            evaluated[node_id] = value
            return value

        parents = tuple(self._run_node(parent, evaluated) for parent in node.parents)
        projected = node.config().get("operator", "").endswith(".project_domain")
        domain = parents[-1].domain if projected else self._resolve_domain(parents)
        self.node_domains[node.logical_id] = domain
        parents = self._admit_materialization(domain, parents)
        key = self._materialization_key(node, parents, domain)
        cacheable = self._contract(node).deterministic and all(parent.cacheable for parent in parents)
        if self.materialization_store is not None and cacheable:
            cached_record = self._loaded_materializations.get(key.identity)
            if cached_record is None and key.identity not in self._pending_materializations:
                lookup = self.materialization_store.query(key)
                cached_record = lookup.materialization if lookup.status == MaterializationStatus.HIT else None
                if lookup.status == MaterializationStatus.PARTIAL:
                    self.persistent_partials += 1
                if cached_record is not None:
                    if cached_record.key != key or not cached_record.panel.domain.equivalent_to(domain):
                        raise ValueError("materialization store returned a different numerical identity or Domain")
                    if cached_record.panel.value_type != node.value_type:
                        raise ValueError("materialization store did not preserve the numerical Node type")
                    expected_traces = tuple(dict.fromkeys(trace for parent in parents for trace in parent.trace_columns))
                    if cached_record.panel.trace_columns != expected_traces:
                        raise ValueError("materialization store did not preserve availability traces")
                    self._loaded_materializations[key.identity] = cached_record
                    replay_node_evidence(node.logical_id, artifacts=cached_record.artifacts,
                        checkpoint=cached_record.checkpoint, training_audits=cached_record.training_audits)
                    self.persistent_hits += 1
                else:
                    self.persistent_misses += 1
            if cached_record is not None:
                self.node_materializations[node.logical_id] = cached_record
                value = self._panel_plan(cached_record.panel)
                value = replace(value, identity=hash_mapping({"execution": key.identity, "name": node.name}),
                                physical_identity=key.identity)
                evaluated[node_id] = value
                return value

        lowered = self._lower_scalar_arithmetic(node, evaluated)
        if lowered is not None:
            lowered = self._record_materialization_plan(node, lowered, key)
            evaluated[node_id] = lowered
            return lowered

        shifted = self._lower_calendar_shift(node, parents)
        if shifted is not None:
            shifted = self._record_materialization_plan(node, shifted, key)
            evaluated[node_id] = shifted
            return shifted
        implicit = self._lower_implicit_dense(node, parents)
        if implicit is not None:
            implicit = self._record_materialization_plan(node, implicit, key)
            evaluated[node_id] = implicit
            return implicit
        contract = self._contract(node)
        prepared = tuple(
            self._ensure_dense(parent)
            if contract.density == InputDensity.DENSE_REQUIRED
            else self._expand_implicit(parent)
            for parent in parents
        )
        if (
            any(parent.trace_columns for parent in prepared)
            and contract.trace_rule == TraceRule.NONE
        ):
            raise ValueError(
                f"operation '{node.name}' has traced inputs but no trace rule"
            )
        identity = hash_mapping(
            {
                "node": node.signature(),
                "parents": [value.identity for value in prepared],
                "domain": domain.signature,
            }
        )
        cacheable = contract.deterministic and all(
            value.cacheable for value in prepared
        )
        cached = self._plan_cache.get(identity) if cacheable else None
        if cached is not None:
            cached = self._record_materialization_plan(node, cached, key)
            evaluated[node_id] = cached
            return cached
        builtin = self._is_builtin_operation(node)
        physical_identity = self._physical_node_identity(
            node,
            prepared,
            domain,
        )
        physical_cached = (
            self._physical_plan_cache.get(physical_identity)
            if (
                cacheable
                and builtin
                and contract.execution == ExecutionMode.LAZY
            )
            else None
        )
        if physical_cached is not None:
            traces = tuple(
                dict.fromkeys(
                    trace
                    for parent in prepared
                    for trace in parent.trace_columns
                )
            )
            value = replace(
                physical_cached,
                identity=identity,
                trace_identity=self._trace_plan_identity(
                    contract,
                    prepared,
                    identity,
                    traces,
                ),
                physical_identity=physical_identity,
            )
            self._plan_cache[identity] = value
            self._diagnostics["semantic_cse_hits"] += 1
            value = self._record_materialization_plan(node, value, key)
            evaluated[node_id] = value
            return value

        inputs = tuple(
            value.frame.select(TIME, ASSET_ID, VALUE) for value in prepared
        )
        eager_result: pl.DataFrame | None = None
        plan_order: str | None = None
        plan_asset_time_ordered = False
        if contract.execution == ExecutionMode.LAZY:
            plan_operation = getattr(
                getattr(node, "operation", None),
                "_plan_operation",
                None,
            )
            if (
                plan_operation is not None
                and len(node.spec_inputs()) == 1
            ):
                if prepared[0].asset_time_ordered:
                    self._diagnostics["sorts_elided"] += 1
                result, plan_order, plan_asset_time_ordered = plan_operation(
                    inputs[0],
                    self._node_parameters(node),
                    prepared[0].order,
                    prepared[0].asset_time_ordered,
                )
            elif (
                plan_operation is not None
                and len(node.spec_inputs()) > 1
                and self._positionally_aligned(prepared)
            ):
                self._diagnostics["positional_operator_hits"] += 1
                common_order = prepared[0].order
                plan_asset_time_ordered = all(
                    parent.asset_time_ordered for parent in prepared
                )
                result, plan_order, plan_asset_time_ordered = plan_operation(
                    inputs,
                    self._node_parameters(node),
                    common_order,
                    plan_asset_time_ordered,
                )
            else:
                result = node.evaluate_frames(*inputs)
            if isinstance(result, pl.DataFrame):
                result = result.lazy()
            if not isinstance(result, pl.LazyFrame):
                raise TypeError(
                    f"Node '{node.name}' returned {type(result)}; "
                    "expected LazyFrame-compatible output"
                )
        else:
            assert self._active_eager_physical_results is not None
            result = (
                self._active_eager_physical_results.get(physical_identity)
                if cacheable and builtin
                else None
            )
            common_subexpression = result is not None
            if result is None:
                if self._defer_eager:
                    raise _DeferredEager(_EagerWork(node, prepared, domain, physical_identity))
                result = self._completed_eager.pop(node_id, None)
            self.eager_barriers += 1
            if result is None:
                eager_inputs = self._collect_eager_inputs(prepared, inputs)
                self.resource_usage["max_parallel_nodes"] = max(self.resource_usage["max_parallel_nodes"], 1)
                with resource_limits(self._effective_limits.for_workers(1)):
                    result = self._compute_eager(node, prepared, eager_inputs, domain)
            if cacheable and builtin:
                self._active_eager_physical_results[physical_identity] = result
            if common_subexpression:
                self._diagnostics["eager_cse_hits"] += 1
            if not isinstance(result, pl.DataFrame):
                raise TypeError(
                    f"Node '{node.name}' returned {type(result)}; "
                    "expected DataFrame at eager barrier"
                )
            eager_result = result
            result = eager_result.lazy()
        if not builtin:
            result = domain.apply_membership_lazy(result)
            self._diagnostics["membership_applications"] += 1

        traces = tuple(
            dict.fromkeys(
                trace
                for parent in prepared
                for trace in parent.trace_columns
            )
        )
        key_identity = self._result_key_identity(
            node,
            prepared,
            identity,
        )
        order = (
            plan_order
            if plan_asset_time_ordered
            else (_TIME_ASSET_ORDER if builtin else None)
        )
        exact_domain = self._result_exact_domain(node, prepared)
        trace_frame, trace_key_identity, trace_order = self._build_trace_plan(
            result,
            prepared,
            contract,
            self._node_parameters(node),
            traces,
            result_key_identity=key_identity,
            result_order=order,
        )
        value = PlanValue(
            frame=result,
            domain=domain,
            density=contract.density,
            identity=identity,
            trace_columns=traces,
            trace_frame=trace_frame,
            trace_identity=self._trace_plan_identity(
                contract, prepared, identity, traces
            ),
            categorical=node.value_type == "category",
            prediction=node.value_type == "prediction",
            cacheable=cacheable,
            key_identity=key_identity,
            order=order,
            trace_key_identity=trace_key_identity,
            trace_order=trace_order,
            exact_domain=exact_domain,
            asset_time_ordered=(
                plan_asset_time_ordered
                or order in {_TIME_ASSET_ORDER, _ASSET_TIME_ORDER}
            ),
            validated_keys=builtin and all(
                parent.validated_keys for parent in prepared
            ),
            physical_identity=physical_identity,
        )
        if eager_result is not None:
            assert self._active_eager_inputs is not None
            self._active_eager_inputs[identity] = eager_result
        if cacheable:
            self._plan_cache[identity] = value
            if builtin and contract.execution == ExecutionMode.LAZY:
                self._physical_plan_cache[physical_identity] = value
        value = self._record_materialization_plan(node, value, key)
        evaluated[node_id] = value
        return value

    def _collect_eager_inputs(
        self,
        prepared: tuple[PlanValue, ...],
        inputs: tuple[pl.LazyFrame, ...],
    ) -> tuple[pl.DataFrame, ...]:
        assert self._active_eager_inputs is not None
        owners: dict[tuple[str, str, str], str] = {}
        missing: list[
            tuple[str, pl.LazyFrame, str]
        ] = []
        seen: set[str] = set()
        for value, frame in zip(prepared, inputs, strict=True):
            layout = (
                (value.domain.signature, value.key_identity, value.order)
                if (
                    value.exact_domain
                    and value.validated_keys
                    and value.key_identity is not None
                    and value.order is not None
                )
                else None
            )
            owner_identity = (
                owners.setdefault(layout, value.identity)
                if layout is not None
                else value.identity
            )
            if (
                value.identity not in self._active_eager_inputs
                and value.identity not in seen
            ):
                seen.add(value.identity)
                missing.append(
                    (
                        value.identity,
                        (
                            frame
                            if owner_identity == value.identity
                            else frame.select(VALUE)
                        ),
                        owner_identity,
                    )
                )
        if missing:
            collected = pl.collect_all([entry[1] for entry in missing])
            self.materializations += 1
            for (identity, _, owner_identity), collected_frame in zip(
                missing, collected, strict=True
            ):
                if identity == owner_identity:
                    self._active_eager_inputs[identity] = collected_frame
            for (identity, _, owner_identity), collected_frame in zip(
                missing, collected, strict=True
            ):
                if identity == owner_identity:
                    continue
                owner = self._active_eager_inputs[owner_identity]
                self._active_eager_inputs[identity] = owner.select(
                    TIME, ASSET_ID
                ).with_columns(collected_frame.get_column(VALUE))
        return tuple(
            self._active_eager_inputs[value.identity]
            for value in prepared
        )

    def _eager_layout(
        self,
        value: PlanValue,
        frame: pl.DataFrame,
    ) -> _EagerLayout:
        if (
            not value.exact_domain
            or not value.validated_keys
            or value.key_identity is None
            or value.order != _TIME_ASSET_ORDER
        ):
            raise ValueError("eager layout requires exact time-asset keys")
        assert self._active_eager_layouts is not None
        key = (
            value.domain.signature,
            value.key_identity,
            value.order,
        )
        cached = self._active_eager_layouts.get(key)
        if cached is not None:
            return cached

        index_dtype = (
            np.uint32
            if len(frame) <= np.iinfo(np.uint32).max
            else np.uint64
        )
        ordered = (
            frame.select(TIME, ASSET_ID)
            .with_row_index("__row")
            .sort([ASSET_ID, TIME], maintain_order=True)
        )
        permutation = ordered.get_column("__row").to_numpy().astype(
            index_dtype,
            copy=False,
        )
        inverse = np.empty(len(frame), dtype=index_dtype)
        inverse[permutation] = np.arange(len(frame), dtype=index_dtype)
        asset_lengths = (
            ordered.group_by(ASSET_ID, maintain_order=True)
            .len()
            .get_column("len")
            .to_numpy()
        )
        time_lengths = (
            frame.group_by(TIME, maintain_order=True)
            .len()
            .get_column("len")
            .to_numpy()
        )
        asset_offsets = np.empty(len(asset_lengths) + 1, dtype=np.int64)
        asset_offsets[0] = 0
        np.cumsum(asset_lengths, dtype=np.int64, out=asset_offsets[1:])
        time_offsets = np.empty(len(time_lengths) + 1, dtype=np.int64)
        time_offsets[0] = 0
        np.cumsum(time_lengths, dtype=np.int64, out=time_offsets[1:])
        layout = _EagerLayout(
            asset_permutation=permutation,
            inverse_permutation=inverse,
            asset_offsets=asset_offsets,
            time_offsets=time_offsets,
        )
        self._active_eager_layouts[key] = layout
        return layout

    def _compute_eager(
        self,
        node: Node,
        prepared: tuple[PlanValue, ...],
        inputs: tuple[pl.DataFrame, ...],
        domain: Domain,
    ) -> pl.DataFrame:
        operation = getattr(
            getattr(node, "operation", None),
            "display_name",
            "",
        )
        regression_operations = {
            "rolling_ols",
            "rolling_ridge",
            "rolling_lasso",
            "rolling_elastic_net",
        }
        if operation == "orthogonalize":
            key_identity = prepared[0].key_identity
            positionally_aligned = (
                key_identity is not None
                and all(parent.exact_domain for parent in prepared)
                and all(parent.validated_keys for parent in prepared)
                and all(
                    parent.key_identity == key_identity
                    and parent.order == _TIME_ASSET_ORDER
                    for parent in prepared
                )
            )
            if positionally_aligned:
                from bagelquant_core.operator.cross_sectional import _orthogonalize_aligned

                if domain.is_dynamic:
                    time_offsets = self._eager_layout(
                        prepared[0],
                        inputs[0],
                    ).time_offsets
                else:
                    asset_count = len(domain.asset_ids)
                    time_offsets = np.arange(
                        0,
                        len(inputs[0]) + asset_count,
                        asset_count,
                        dtype=np.int64,
                    )
                return _orthogonalize_aligned(
                    inputs,
                    fit_intercept=bool(
                        self._node_parameters(node).get(
                            "fit_intercept", False
                        )
                    ),
                    group_offsets=time_offsets,
                )
            return node.evaluate_frames(*inputs)
        if operation in regression_operations:
            key_identity = prepared[0].key_identity
            positionally_aligned = (
                key_identity is not None
                and all(parent.exact_domain for parent in prepared)
                and all(parent.validated_keys for parent in prepared)
                and all(
                    parent.key_identity == key_identity
                    and parent.order == _TIME_ASSET_ORDER
                    for parent in prepared
                )
            )
            if positionally_aligned:
                from bagelquant_core.operator.regression import _rolling_regression_aligned

                static_shape = (
                    (len(domain.times), len(domain.asset_ids))
                    if not domain.is_dynamic
                    else None
                )
                layout = (
                    self._eager_layout(prepared[0], inputs[0])
                    if domain.is_dynamic
                    else None
                )
                return _rolling_regression_aligned(
                    inputs,
                    operation=operation,
                    config=self._node_parameters(node),
                    static_shape=static_shape,
                    asset_permutation=(
                        None
                        if layout is None
                        else layout.asset_permutation
                    ),
                    group_offsets=(
                        None
                        if layout is None
                        else layout.asset_offsets
                    ),
                    diagnostics=self._diagnostics,
                )
            return node.evaluate_frames(*inputs)
        if operation not in {"rolling_rank", "rolling_percentile"}:
            return node.evaluate_frames(*inputs)

        from bagelquant_core.operator.rolling_stats import (
            _rolling_last_rank_pair,
            _validate_window,
        )

        parameters = self._node_parameters(node)
        window = parameters.get("window")
        if not isinstance(window, int) or isinstance(window, bool):
            return node.evaluate_frames(*inputs)
        min_periods = _validate_window(
            window,
            parameters.get("min_periods"),
        )
        parent = prepared[0]
        key = (
            "rolling_rank_pair",
            parent.identity,
            domain.signature,
            window,
            min_periods,
        )
        assert self._active_eager_results is not None
        pair = self._active_eager_results.get(key)
        if pair is None:
            static_shape = (
                (len(domain.times), len(domain.asset_ids))
                if (
                    not domain.is_dynamic
                    and parent.exact_domain
                    and parent.validated_keys
                    and parent.order == _TIME_ASSET_ORDER
                )
                else None
            )
            layout = (
                self._eager_layout(parent, inputs[0])
                if static_shape is None
                and parent.exact_domain
                and parent.validated_keys
                and parent.order == _TIME_ASSET_ORDER
                else None
            )
            pair = _rolling_last_rank_pair(
                inputs[0],
                window=window,
                min_periods=min_periods,
                static_shape=static_shape,
                asset_permutation=(
                    None
                    if layout is None
                    else layout.asset_permutation
                ),
                group_offsets=(
                    None
                    if layout is None
                    else layout.asset_offsets
                ),
            )
            self._active_eager_results[key] = pair
        return pair[0] if operation == "rolling_rank" else pair[1]

    def _materialize_node(
        self,
        node: Node,
        plan: PlanValue,
        *,
        dense_output: bool,
    ) -> Node:
        cached = self.cache.get(plan.identity) if plan.cacheable else None
        if cached is not None:
            logger.debug("Cache hit: %s", node.name)
            return cached
        panel_type = Node
        output = panel_type._from_plan(
            self._frame_with_traces(plan),
            value_type=node.value_type,
            domain=plan.domain,
            name=node.name,
            metadata=node.metadata,
            identity=plan.identity,
            trace_identity=plan.trace_identity,
            trace_columns=plan.trace_columns,
            dense_output=dense_output,
            validated_keys=plan.validated_keys,
            exact_domain=plan.exact_domain,
            key_identity=plan.key_identity,
        )
        if dense_output:
            self.materializations += 1
            if plan.cacheable:
                self._remember_output(plan.identity, output)
        return output

    def _lower_scalar_arithmetic(
        self,
        node: Node,
        evaluated: dict[int, PlanValue],
    ) -> PlanValue | None:
        """Fuse ``constant(panel)`` into a binary arithmetic expression."""

        if len(node.spec_inputs()) < 2 or len(node.parents) != 2:
            return None
        operation = getattr(getattr(node, "operation", None), "display_name", "")
        expressions = {
            "add": lambda column, scalar, _constant_first: column + scalar,
            "mul": lambda column, scalar, _constant_first: column * scalar,
            "sub": lambda column, scalar, constant_first: (
                scalar - column if constant_first else column - scalar
            ),
            "div": lambda column, scalar, constant_first: (
                scalar / column if constant_first else column / scalar
            ),
        }
        if operation not in expressions:
            return None
        constant_index = next(
            (
                index
                for index, parent in enumerate(node.parents)
                if parent.node_type == "operator" and len(parent.spec_inputs()) == 1
                and getattr(
                    getattr(parent, "operation", None), "display_name", ""
                )
                == "constant"
            ),
            None,
        )
        if constant_index is None:
            return None

        constant_node = node.parents[constant_index]
        other_node = node.parents[1 - constant_index]
        scalar = self._node_parameters(constant_node).get("value", 1)
        if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
            return None
        constant_source = self._run_node(constant_node.parents[0], evaluated)
        other = self._run_node(other_node, evaluated)
        domain = self._resolve_domain((constant_source, other))
        contract = self._contract(node)
        cacheable = (
            contract.deterministic
            and constant_source.cacheable
            and other.cacheable
        )
        identity = hash_mapping(
            {
                "lowered_scalar": node.signature(),
                "scalar": float(scalar),
                "constant_source": constant_source.identity,
                "other": other.identity,
                "domain": domain.signature,
            }
        )
        cached = self._plan_cache.get(identity) if cacheable else None
        if cached is not None:
            return cached

        result = other.frame.select(
            TIME,
            ASSET_ID,
            expressions[operation](
                pl.col(VALUE), pl.lit(float(scalar)), constant_index == 0
            ).alias(VALUE),
        )
        traces = tuple(
            dict.fromkeys(
                (*constant_source.trace_columns, *other.trace_columns)
            )
        )
        key_identity = other.key_identity
        trace_frame, trace_key_identity, trace_order = self._build_trace_plan(
            result,
            (constant_source, other),
            contract,
            self._node_parameters(node),
            traces,
            result_key_identity=key_identity,
            result_order=other.order,
        )
        value = PlanValue(
            frame=result,
            domain=domain,
            density=InputDensity.SPARSE_OK,
            identity=identity,
            trace_columns=traces,
            trace_frame=trace_frame,
            trace_identity=self._trace_plan_identity(
                contract,
                (constant_source, other),
                identity,
                traces,
            ),
            categorical=False,
            prediction=constant_source.prediction and other.prediction,
            cacheable=cacheable,
            key_identity=key_identity,
            order=other.order,
            trace_key_identity=trace_key_identity,
            trace_order=trace_order,
            exact_domain=other.exact_domain,
            asset_time_ordered=other.asset_time_ordered,
            validated_keys=(
                constant_source.validated_keys and other.validated_keys
            ),
        )
        if cacheable:
            self._plan_cache[identity] = value
        return value

    def _lower_implicit_dense(
        self,
        node: Node,
        parents: tuple[PlanValue, ...],
    ) -> PlanValue | None:
        """Represent pointwise dense defaults without constructing a grid."""

        operation = getattr(getattr(node, "operation", None), "display_name", "")
        contract = self._contract(node)
        if len(node.spec_inputs()) == 1 and operation in {
            "constant",
            "fillna",
            "fillna_zero",
        }:
            parent = parents[0]
            parameters = self._node_parameters(node)
            scalar = float(
                parameters.get("value", 0.0 if operation != "constant" else 1.0)
            )
            if operation == "constant":
                expression = pl.lit(scalar)
                default = scalar
            else:
                expression = pl.col(VALUE).fill_null(scalar).fill_nan(scalar)
                default = (
                    scalar
                    if parent.default_value is None
                    else parent.default_value
                )
            identity = hash_mapping(
                {
                    "implicit_dense": node.signature(),
                    "parent": parent.identity,
                    "domain": parent.domain.signature,
                }
            )
            cacheable = contract.deterministic and parent.cacheable
            cached = self._plan_cache.get(identity) if cacheable else None
            if cached is not None:
                return cached
            frame = parent.frame.select(
                TIME, ASSET_ID, expression.alias(VALUE)
            )
            traces = parent.trace_columns
            trace_frame, trace_key_identity, trace_order = self._build_trace_plan(
                frame,
                (parent,),
                contract,
                parameters,
                traces,
                result_key_identity=parent.key_identity,
                result_order=parent.order,
            )
            value = PlanValue(
                frame=frame,
                domain=parent.domain,
                density=InputDensity.DENSE_REQUIRED,
                identity=identity,
                trace_columns=traces,
                trace_frame=trace_frame,
                trace_identity=self._trace_plan_identity(
                    contract, (parent,), identity, traces
                ),
                default_value=default,
                categorical=parent.categorical,
                prediction=parent.prediction,
                cacheable=cacheable,
                key_identity=parent.key_identity,
                order=parent.order,
                trace_key_identity=trace_key_identity,
                trace_order=trace_order,
                exact_domain=parent.exact_domain,
                asset_time_ordered=parent.asset_time_ordered,
                validated_keys=parent.validated_keys,
            )
            if cacheable:
                self._plan_cache[identity] = value
            return value

        reducers = {
            "add": lambda values: values[0] + values[1],
            "sub": lambda values: values[0] - values[1],
            "mul": lambda values: values[0] * values[1],
            "div": lambda values: values[0] / values[1],
            "sum_frames": lambda values: sum(values[1:], values[0]),
            "mean": lambda values: sum(values[1:], values[0]) / len(values),
            "product": lambda values: pl.fold(
                acc=pl.lit(1.0),
                function=lambda acc, value: acc * value,
                exprs=values,
            ),
        }
        if (
            len(node.spec_inputs()) < 2
            or operation not in reducers
            or not any(parent.default_value is not None for parent in parents)
        ):
            return None
        domain = self._resolve_domain(parents)
        identity = hash_mapping(
            {
                "implicit_operator": node.signature(),
                "parents": [parent.identity for parent in parents],
                "domain": domain.signature,
            }
        )
        cacheable = contract.deterministic and all(
            parent.cacheable for parent in parents
        )
        cached = self._plan_cache.get(identity) if cacheable else None
        if cached is not None:
            return cached

        joined = parents[0].frame.rename({VALUE: "__value_0"})
        expressions = [
            self._implicit_column("__value_0", parents[0].default_value)
        ]
        for index, parent in enumerate(parents[1:], start=1):
            name = f"__value_{index}"
            joined = joined.join(
                parent.frame.rename({VALUE: name}),
                on=[TIME, ASSET_ID],
                how="full",
                coalesce=True,
            )
            expressions.append(
                self._implicit_column(name, parent.default_value)
            )
        result = joined.select(
            TIME,
            ASSET_ID,
            reducers[operation](expressions).alias(VALUE),
        )
        defaults = [parent.default_value for parent in parents]
        default = self._reduce_defaults(operation, defaults)
        traces = tuple(
            dict.fromkeys(
                trace
                for parent in parents
                for trace in parent.trace_columns
            )
        )
        key_identity = self._combined_key_identity(parents, identity)
        trace_frame, trace_key_identity, trace_order = self._build_trace_plan(
            result,
            parents,
            contract,
            self._node_parameters(node),
            traces,
            result_key_identity=key_identity,
            result_order=None,
        )
        value = PlanValue(
            frame=result,
            domain=domain,
            density=(
                InputDensity.DENSE_REQUIRED
                if default is not None
                else InputDensity.SPARSE_OK
            ),
            identity=identity,
            trace_columns=traces,
            trace_frame=trace_frame,
            trace_identity=self._trace_plan_identity(
                contract, parents, identity, traces
            ),
            default_value=default,
            prediction=all(parent.prediction for parent in parents),
            cacheable=cacheable,
            key_identity=key_identity,
            trace_key_identity=trace_key_identity,
            trace_order=trace_order,
            exact_domain=(
                default is not None
                and all(parent.exact_domain for parent in parents)
            ),
            asset_time_ordered=False,
            validated_keys=all(parent.validated_keys for parent in parents),
        )
        if cacheable:
            self._plan_cache[identity] = value
        return value

    def _lower_calendar_shift(
        self,
        node: Node,
        parents: tuple[PlanValue, ...],
    ) -> PlanValue | None:
        """Lower lag to a sparse calendar-key shift instead of a dense grid."""

        operation = getattr(getattr(node, "operation", None), "display_name", "")
        if len(node.spec_inputs()) != 1 or operation != "lag":
            return None
        parent = parents[0]
        periods = self._node_parameters(node).get("periods", 1)
        if not isinstance(periods, int) or isinstance(periods, bool):
            raise TypeError("lag periods must be an integer")
        if periods <= 0:
            raise ValueError("lag periods must be positive")
        times = parent.domain.times
        if periods >= len(times):
            mapping = pl.DataFrame(
                schema={TIME: pl.Date, "__shifted_time": pl.Date}
            ).lazy()
        else:
            mapping = pl.DataFrame(
                {
                    TIME: times.head(len(times) - periods),
                    "__shifted_time": times.tail(len(times) - periods),
                }
            ).lazy()

        def shift(frame: pl.LazyFrame, columns: tuple[str, ...]) -> pl.LazyFrame:
            return (
                frame.join(mapping, on=TIME, how="inner")
                .select(
                    pl.col("__shifted_time").alias(TIME),
                    ASSET_ID,
                    *columns,
                )
                .sort([TIME, ASSET_ID])
            )

        identity = hash_mapping(
            {
                "calendar_shift": node.signature(),
                "parent": parent.identity,
                "domain": parent.domain.signature,
            }
        )
        contract = self._contract(node)
        cacheable = contract.deterministic and parent.cacheable
        cached = self._plan_cache.get(identity) if cacheable else None
        if cached is not None:
            return cached
        value = PlanValue(
            frame=shift(parent.frame, (VALUE,)),
            domain=parent.domain,
            density=InputDensity.SPARSE_OK,
            identity=identity,
            trace_columns=parent.trace_columns,
            trace_frame=(
                shift(parent.trace_frame, parent.trace_columns)
                if parent.trace_frame is not None
                else None
            ),
            trace_identity=(
                hash_mapping(
                    {
                        "shift_trace": parent.trace_identity,
                        "periods": periods,
                        "domain": parent.domain.signature,
                    }
                )
                if parent.trace_identity is not None
                else None
            ),
            default_value=parent.default_value,
            categorical=parent.categorical,
            prediction=parent.prediction,
            cacheable=cacheable,
            key_identity=identity,
            order=_TIME_ASSET_ORDER,
            trace_key_identity=identity,
            trace_order=_TIME_ASSET_ORDER,
            exact_domain=False,
            asset_time_ordered=True,
            validated_keys=parent.validated_keys,
        )
        if cacheable:
            self._plan_cache[identity] = value
        return value

    @staticmethod
    def _implicit_column(name: str, default: float | None) -> pl.Expr:
        column = pl.col(name)
        return column if default is None else column.fill_null(default)

    @staticmethod
    def _reduce_defaults(
        operation: str,
        values: list[float | None],
    ) -> float | None:
        if any(value is None for value in values):
            return None
        numeric = [float(value) for value in values if value is not None]
        if operation in {"add", "sum_frames"}:
            return sum(numeric)
        if operation == "sub":
            return numeric[0] - numeric[1]
        if operation in {"mul", "product"}:
            result = 1.0
            for value in numeric:
                result *= value
            return result
        if operation == "div":
            return numeric[0] / numeric[1]
        if operation == "mean":
            return sum(numeric) / len(numeric)
        return None

    def _ensure_dense(self, value: PlanValue) -> PlanValue:
        if value.exact_domain and value.default_value is None:
            return value
        if value.default_value is not None:
            return self._expand_implicit(value)
        identity = hash_mapping(
            {"dense": value.identity, "domain": value.domain.signature}
        )
        cached = self._plan_cache.get(identity)
        if cached is not None:
            return cached
        result = PlanValue(
            frame=value.domain.align_lazy(value.frame),
            domain=value.domain,
            density=InputDensity.DENSE_REQUIRED,
            identity=identity,
            trace_columns=value.trace_columns,
            trace_frame=(
                value.domain.grid_lazy().join(
                    value.trace_frame,
                    on=[TIME, ASSET_ID],
                    how="left",
                    maintain_order="left",
                )
                if value.trace_frame is not None
                else None
            ),
            trace_identity=value.trace_identity,
            categorical=value.categorical,
            prediction=value.prediction,
            cacheable=value.cacheable,
            key_identity=f"domain:{value.domain.signature}",
            order=_TIME_ASSET_ORDER,
            trace_key_identity=f"domain:{value.domain.signature}",
            trace_order=_TIME_ASSET_ORDER,
            exact_domain=True,
            asset_time_ordered=True,
            validated_keys=value.validated_keys,
            physical_identity=value.physical_identity,
        )
        self._plan_cache[identity] = result
        return result

    def _expand_implicit(self, value: PlanValue) -> PlanValue:
        if value.default_value is None:
            return value
        identity = hash_mapping(
            {
                "expanded_default": value.identity,
                "default": value.default_value,
                "domain": value.domain.signature,
            }
        )
        cached = self._plan_cache.get(identity) if value.cacheable else None
        if cached is not None:
            return cached
        frame = value.domain.align_lazy(value.frame).with_columns(
            pl.col(VALUE).fill_null(value.default_value)
        )
        result = PlanValue(
            frame=frame,
            domain=value.domain,
            density=InputDensity.DENSE_REQUIRED,
            identity=identity,
            trace_columns=value.trace_columns,
            trace_frame=(
                value.domain.grid_lazy().join(
                    value.trace_frame,
                    on=[TIME, ASSET_ID],
                    how="left",
                    maintain_order="left",
                )
                if value.trace_frame is not None
                else None
            ),
            trace_identity=value.trace_identity,
            categorical=value.categorical,
            prediction=value.prediction,
            cacheable=value.cacheable,
            key_identity=f"domain:{value.domain.signature}",
            order=_TIME_ASSET_ORDER,
            trace_key_identity=f"domain:{value.domain.signature}",
            trace_order=_TIME_ASSET_ORDER,
            exact_domain=True,
            asset_time_ordered=True,
            validated_keys=value.validated_keys,
            physical_identity=value.physical_identity,
        )
        if value.cacheable:
            self._plan_cache[identity] = result
        return result

    def _build_trace_plan(
        self,
        result: pl.LazyFrame,
        parents: tuple[PlanValue, ...],
        contract: OperationContract,
        config: dict[str, Any],
        traces: tuple[str, ...],
        *,
        result_key_identity: str | None,
        result_order: str | None,
    ) -> tuple[pl.LazyFrame | None, str | None, str | None]:
        if not traces:
            return None, None, None
        if contract.trace_rule == TraceRule.NONE:
            raise ValueError(
                "operation with traced inputs must declare a trace rule"
            )
        if contract.trace_rule == TraceRule.CUSTOM:
            assert contract.trace_function is not None
            return (
                contract.trace_function(
                    tuple(self._frame_with_traces(value) for value in parents),
                    result,
                    config,
                    traces,
                ),
                result_key_identity,
                None,
            )

        keys = result.select(TIME, ASSET_ID)
        return self._trace_plan(
            keys,
            parents,
            contract.trace_rule,
            config,
            traces,
            result_key_identity=result_key_identity,
            result_order=result_order,
        )

    def _trace_plan(
        self,
        keys: pl.LazyFrame,
        parents: tuple[PlanValue, ...],
        rule: TraceRule,
        config: dict[str, Any],
        traces: tuple[str, ...],
        *,
        result_key_identity: str | None,
        result_order: str | None,
    ) -> tuple[pl.LazyFrame, str | None, str | None]:
        if rule == TraceRule.PARENT_MAX:
            return self._parent_max_traces(
                keys,
                parents,
                traces,
                result_key_identity=result_key_identity,
                result_order=result_order,
            )

        parent = parents[0]
        available = [
            trace for trace in traces if trace in parent.trace_columns
        ]
        assert parent.trace_frame is not None
        base = parent.trace_frame.select(TIME, ASSET_ID, *available)
        needs_asset_time = rule != TraceRule.PASSTHROUGH
        trace_time_ordered = parent.trace_order in {
            _TIME_ASSET_ORDER,
            _ASSET_TIME_ORDER,
        }
        if needs_asset_time and not trace_time_ordered:
            base = base.sort([ASSET_ID, TIME])
        elif needs_asset_time:
            self._diagnostics["sorts_elided"] += 1
        transformed_order = (
            (
                parent.trace_order
                if trace_time_ordered
                else _ASSET_TIME_ORDER
            )
            if needs_asset_time
            else parent.trace_order
        )
        if rule == TraceRule.PASSTHROUGH:
            transformed = base
        elif rule == TraceRule.SHIFT:
            periods = int(config.get("periods", 1))
            transformed = base.with_columns(
                pl.col(trace).shift(periods).over(ASSET_ID)
                for trace in available
            )
        elif rule == TraceRule.CURRENT_AND_SHIFT_MAX:
            periods = int(config.get("periods", config.get("interval", 1)))
            expressions = []
            for trace in available:
                prior = pl.col(trace).shift(periods).over(ASSET_ID)
                expressions.append(
                    pl.when(pl.col(trace).is_not_null())
                    .then(pl.max_horizontal(pl.col(trace), prior))
                    .otherwise(None)
                    .alias(trace)
                )
            transformed = base.with_columns(expressions)
        elif rule == TraceRule.ROLLING_MAX and "window" in config:
            window = int(config["window"])
            transformed = base.with_columns(
                pl.col(trace)
                .cast(pl.Int32)
                .rolling_max(window_size=window, min_samples=1)
                .over(ASSET_ID)
                .cast(pl.Date)
                for trace in available
            )
        elif rule == TraceRule.CUMULATIVE_MAX:
            transformed = base.with_columns(
                pl.col(trace)
                .cast(pl.Int32)
                .cum_max()
                .over(ASSET_ID)
                .cast(pl.Date)
                for trace in available
            )
        elif rule in {TraceRule.FORWARD_FILL, TraceRule.BACKWARD_FILL}:
            strategy = (
                "forward"
                if rule == TraceRule.FORWARD_FILL
                else "backward"
            )
            limit = config.get("limit")
            transformed = base.with_columns(
                pl.col(trace)
                .fill_null(strategy=strategy, limit=limit)
                .over(ASSET_ID)
                for trace in available
            )
        else:
            transformed = base
        if (
            result_key_identity is not None
            and result_key_identity == parent.trace_key_identity
        ):
            return (
                transformed.select(TIME, ASSET_ID, *available),
                result_key_identity,
                transformed_order,
            )
        return (
            keys.join(
                transformed,
                on=[TIME, ASSET_ID],
                how="left",
                maintain_order="left",
            ),
            result_key_identity,
            result_order,
        )

    @staticmethod
    def _parent_max_traces(
        keys: pl.LazyFrame,
        parents: tuple[PlanValue, ...],
        traces: tuple[str, ...],
        *,
        result_key_identity: str | None,
        result_order: str | None,
    ) -> tuple[pl.LazyFrame, str | None, str | None]:
        traced = [
            parent for parent in parents if parent.trace_frame is not None
        ]
        if (
            traced
            and traced[0].trace_identity is not None
            and all(
                parent.trace_identity == traced[0].trace_identity
                for parent in traced[1:]
            )
            and all(
                trace in traced[0].trace_columns for trace in traces
            )
            and result_key_identity is not None
            and result_key_identity == traced[0].trace_key_identity
        ):
            assert traced[0].trace_frame is not None
            return (
                traced[0].trace_frame.select(
                    TIME,
                    ASSET_ID,
                    *traces,
                ),
                result_key_identity,
                traced[0].trace_order,
            )
        frame = keys
        output_expressions: list[pl.Expr] = []
        names_by_trace: dict[str, list[str]] = {
            trace: [] for trace in traces
        }
        for index, parent in enumerate(parents):
            if parent.trace_frame is None:
                continue
            expressions: list[pl.Expr] = []
            for trace in traces:
                if trace in parent.trace_columns:
                    name = f"__trace_{trace}_{index}"
                    names_by_trace[trace].append(name)
                    expressions.append(pl.col(trace).alias(name))
            if expressions:
                frame = frame.join(
                    parent.trace_frame.select(
                        TIME, ASSET_ID, *expressions
                    ),
                    on=[TIME, ASSET_ID],
                    how="left",
                )
        for trace in traces:
            names = names_by_trace[trace]
            if names:
                output_expressions.append(
                    pl.max_horizontal(*names).alias(trace)
                )
        return (
            frame.with_columns(output_expressions).select(
                TIME,
                ASSET_ID,
                *traces,
            ),
            result_key_identity,
            result_order,
        )

    def _frame_with_traces(self, plan: PlanValue) -> pl.LazyFrame:
        plan = self._expand_implicit(plan)
        if plan.trace_frame is None:
            return plan.frame
        return plan.frame.join(
            plan.trace_frame,
            on=[TIME, ASSET_ID],
            how="left",
        )

    def plan(self, graph: Graph) -> tuple[Node, ...]:
        graph.validate()
        return graph.topological_sort()

    def clear_cache(self) -> None:
        self.cache.clear()
        self._plan_cache.clear()
        self._physical_plan_cache.clear()

    @staticmethod
    def _is_builtin_operation(node: Node) -> bool:
        operation = getattr(node, "operation", None)
        function = getattr(operation, "operation", None)
        module = getattr(function, "__module__", "")
        return module.startswith("bagelquant_core.")

    def _result_key_identity(
        self,
        node: Node,
        parents: tuple[PlanValue, ...],
        node_identity: str,
    ) -> str | None:
        if (
            len(node.spec_inputs()) == 1
            and self._is_builtin_operation(node)
        ):
            if node.config().get("operator", "").endswith((".project_domain", ".rebalance")):
                return hash_mapping({"selected_keys": node_identity})
            return parents[0].key_identity
        return self._combined_key_identity(parents, node_identity)

    def _physical_node_identity(
        self,
        node: Node,
        parents: tuple[PlanValue, ...],
        domain: Domain,
    ) -> str:
        return self._materialization_key(node, parents, domain).identity

    def _result_exact_domain(
        self,
        node: Node,
        parents: tuple[PlanValue, ...],
    ) -> bool:
        if (
            not parents
            or not self._is_builtin_operation(node)
            or node._registered.factory is not None
        ):
            return False
        if len(node.spec_inputs()) == 1:
            if node.config().get("operator", "").endswith((".rebalance", ".project_domain")):
                return False
            return parents[0].exact_domain
        key_identity = parents[0].key_identity
        return (
            key_identity is not None
            and all(parent.exact_domain for parent in parents)
            and all(parent.key_identity == key_identity for parent in parents)
        )

    @staticmethod
    def _positionally_aligned(parents: tuple[PlanValue, ...]) -> bool:
        if not parents:
            return False
        key_identity = parents[0].key_identity
        order = parents[0].order
        return (
            key_identity is not None
            and order is not None
            and all(parent.exact_domain for parent in parents)
            and all(parent.validated_keys for parent in parents)
            and all(
                parent.key_identity == key_identity
                and parent.order == order
                for parent in parents
            )
        )

    @staticmethod
    def _combined_key_identity(
        parents: tuple[PlanValue, ...],
        node_identity: str,
    ) -> str:
        identities = [parent.key_identity for parent in parents]
        if identities and identities[0] is not None and len(set(identities)) == 1:
            return identities[0]
        return hash_mapping(
            {
                "result_keys": node_identity,
                "parents": identities,
            }
        )

    @staticmethod
    def _trace_plan_identity(
        contract: OperationContract,
        parents: tuple[PlanValue, ...],
        node_identity: str,
        traces: tuple[str, ...],
    ) -> str | None:
        if not traces:
            return None
        identities = [
            parent.trace_identity
            for parent in parents
            if parent.trace_identity is not None
        ]
        if (
            contract.trace_rule == TraceRule.PASSTHROUGH
            and len(identities) == 1
        ):
            return identities[0]
        if (
            contract.trace_rule == TraceRule.PARENT_MAX
            and identities
            and len(set(identities)) == 1
        ):
            return identities[0]
        return hash_mapping(
            {
                "trace_node": node_identity,
                "rule": contract.trace_rule.value,
                "parents": identities,
            }
        )

    @staticmethod
    def _resolve_domain(inputs: tuple[PlanValue, ...]) -> Domain:
        if not inputs:
            raise ValueError("Derived nodes require at least one panel input")
        domain = inputs[0].domain
        if any(not domain.equivalent_to(value.domain) for value in inputs[1:]):
            raise ValueError("Operator inputs must use equivalent Domains")
        return domain

    @staticmethod
    def _contract(node: Node) -> OperationContract:
        contract = getattr(node, "contract", None)
        if not isinstance(contract, OperationContract):
            raise TypeError(f"Node '{node.name}' has no operation contract")
        return contract

    @staticmethod
    def _node_parameters(node: Node) -> dict[str, Any]:
        config = dict(node.config())
        config.pop("operator", None)
        return config


_ExecutionRuntime = ExecutionRuntime

__all__ = ["ExecutionRuntime", "PlanValue"]
