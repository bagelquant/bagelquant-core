"""Immutable canonical-kernel work and results, separate from publication.

The coordinator owns temporary Arrow files, frozen proofs and all CoreStore
writes. A worker consumes only explicit local staging paths and computes the
same finite kernel as inline execution. It never opens a numerical store.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import polars as pl

from .domain import Domain
from .execution import ExecutionRuntime
from .logical import LogicalGraphSpec
from .node import Node
from .resources import ResourceLimits, active_resource_limits, resource_limits


@dataclass(frozen=True, slots=True)
class FiniteKernelInput:
    node_id: str
    input_key: str | None
    frame_file: str
    membership_file: str | None
    calendar: tuple[date, ...]
    assets: tuple[str, ...]
    domain_identity: str
    identity: str
    value_type: str
    trace_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FiniteKernelWork:
    staging_root: str
    node_id: str
    spec_json: str
    parameters_json: str
    inputs: tuple[FiniteKernelInput, ...]
    output_start: date
    limits: ResourceLimits
    estimated_payload_bytes: int
    output_nodes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Return a JSON-compatible descriptor, containing no numerical body."""
        body = asdict(self)
        body["output_start"] = self.output_start.isoformat()
        for item in body["inputs"]:
            item["calendar"] = [day.isoformat() for day in item["calendar"]]
        return body

    @classmethod
    def from_dict(cls, body: dict) -> FiniteKernelWork:
        """Decode an explicit trusted coordinator descriptor."""
        fields = dict(body)
        fields["output_start"] = date.fromisoformat(fields["output_start"])
        fields["limits"] = ResourceLimits(**fields["limits"])
        fields["output_nodes"] = tuple(fields.get("output_nodes", ()))
        fields["inputs"] = tuple(FiniteKernelInput(**{**item,
            "calendar": tuple(date.fromisoformat(day) for day in item["calendar"]),
            "assets": tuple(item["assets"]), "trace_columns": tuple(item["trace_columns"])})
            for item in fields["inputs"])
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class FiniteKernelResult:
    node_id: str
    frame_file: str
    value_type: str
    trace_columns: tuple[str, ...]
    rows: int
    intermediates: tuple[tuple[str, str, str, tuple[str, ...], int], ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, body: dict) -> FiniteKernelResult:
        return cls(**{**body, "trace_columns": tuple(body["trace_columns"]),
            "intermediates": tuple((node, filename, value_type, tuple(traces), rows)
                for node, filename, value_type, traces, rows in body.get("intermediates", ()))})


KernelDispatch = Callable[[FiniteKernelWork], FiniteKernelResult]


def finite_payload_bytes(frames) -> int:
    """Charge every required frame from its immutable coordinate metadata.

    Keys and trace buffers may share storage, but admission does not depend on
    that optimizer choice. Retained frames also reserve fixed object overhead.
    No numerical values, source windows or staged files are read here.
    """
    return sum(1024 + domain.size * (32 + max(8, domain.asset_ids.str.len_bytes().max() or 0)
        + 8 * len(traces)) for domain, traces in frames)


def admit_finite_payload(payload_bytes: int, limits: ResourceLimits | None = None) -> None:
    """Reserve retained frames plus eight copies/temporary working sets."""
    selected = limits or active_resource_limits()
    working = payload_bytes * 8
    available = selected.memory_target_mib * 1024 * 1024
    if working > available:
        raise MemoryError(f"Finite kernel or region needs an estimated {working} bytes for retained outputs, inputs "
            f"and temporary copies; its admitted memory is {available} bytes. Increase admitted memory "
            "or request explicit persist_nodes boundaries before initializing a fresh graph; "
            "completed history is retained and is never replayed to insert boundaries.")


def _staged(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("finite work file must be a direct staging child")
    path = root / filename
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("finite work file escapes its staging root")
    return path


def _run_finite_kernel(spec, node_id, scoped, parameters, output_start, check_canceled):
    return _run_finite_outputs(spec, (node_id,), scoped, parameters, output_start, check_canceled)[node_id]


def _run_finite_outputs(spec, output_nodes, scoped, parameters, output_start, check_canceled):
    from .graph import Graph
    declared = {node.node_id: node for node in spec.nodes}
    bindings = {declared[parent].input_key: value for parent, value in scoped.items()
        if declared[parent].node_type == "input"}
    retained = {parent: value for parent, value in scoped.items() if declared[parent].node_type == "operator"}
    local = LogicalGraphSpec({node_id: node_id for node_id in output_nodes}, spec.nodes)
    bound = Graph.from_logical_spec(local, inputs=bindings, node_bindings=retained, parameter_bindings=parameters)
    runtime = ExecutionRuntime(check_canceled=check_canceled)
    result = runtime.run(bound, dense_output=False)
    if runtime.node_artifacts or runtime.node_checkpoints or runtime.node_training_audits:
        raise ValueError("finite contract cannot discard operator evidence")
    results = result if isinstance(result, dict) else {output_nodes[0]: result}
    outputs = {}
    for node, value in results.items():
        frame = value.collect(dense=False, include_traces=True)
        if output_start > value.domain.times.min():
            frame = frame.filter(pl.col("time") >= output_start)
        outputs[node] = frame, value.value_type, value.trace_columns
    return outputs


def compute_finite_work(work: FiniteKernelWork) -> FiniteKernelResult:
    """Compute a single frozen canonical kernel; never publish or open CoreStore.

    The caller configures native pools before importing Core in a fresh worker.
    Resource limits describe only its admitted local share. Cancellation and
    closing revision fences remain with the coordinator before publication.
    """
    if (isinstance(work.estimated_payload_bytes, bool) or not isinstance(work.estimated_payload_bytes, int)
            or work.estimated_payload_bytes <= 0):
        raise ValueError("finite work requires its original positive retained-payload estimate")
    admit_finite_payload(work.estimated_payload_bytes, work.limits)
    root = Path(work.staging_root)
    spec = LogicalGraphSpec.from_dict(json.loads(work.spec_json))
    declared = {node.node_id: node for node in spec.nodes}
    if work.node_id not in declared:
        raise ValueError("finite work has no declared output")
    scoped = {}
    for item in work.inputs:
        if item.node_id not in declared or declared[item.node_id].input_key != item.input_key:
            raise ValueError("finite input differs from its logical declaration")
        universe = (item.assets if item.membership_file is None else
            pl.read_ipc(_staged(root, item.membership_file), memory_map=False).with_columns(pl.lit(True).alias("active")))
        domain = Domain(calendar=item.calendar, universe=universe)
        if domain.signature != item.domain_identity:
            raise ValueError("finite input Domain differs from frozen coordinates")
        frame = pl.read_ipc(_staged(root, item.frame_file), memory_map=False)
        scoped[item.node_id] = Node.from_domain(frame, domain, identity=item.identity,
            value_type=item.value_type, trace_columns=item.trace_columns,
            trace_identity=item.identity if item.trace_columns else None)
    with resource_limits(work.limits):
        outputs = _run_finite_outputs(spec, work.output_nodes or (work.node_id,), scoped,
            json.loads(work.parameters_json), work.output_start, lambda: None)
    frame, value_type, traces = outputs[work.node_id]
    destination = _staged(root, "result.arrow")
    if destination.exists():
        raise ValueError("finite result already exists")
    pending = _staged(root, "result.partial")
    frame.write_ipc(pending)
    pending.replace(destination)
    intermediates = []
    for index, (node, (values, kind, columns)) in enumerate(outputs.items()):
        if node != work.node_id:
            filename = f"intermediate-{index}.arrow"
            values.write_ipc(_staged(root, filename))
            intermediates.append((node, filename, kind.value, columns, values.height))
    return FiniteKernelResult(work.node_id, destination.name, value_type.value, traces, frame.height, tuple(intermediates))


def dispatch_finite_kernel(spec, node_id, scoped, parameters, output_start, dispatch, check_canceled,
                           *, estimated_payload_bytes):
    """Stage bounded typed work and consume its result before files expire."""
    return dispatch_finite_outputs(spec, node_id, (node_id,), scoped, parameters, output_start, dispatch, check_canceled,
        estimated_payload_bytes=estimated_payload_bytes)[node_id]


def dispatch_finite_outputs(spec, node_id, output_nodes, scoped, parameters, output_start, dispatch, check_canceled,
                            *, estimated_payload_bytes):
    from .preparation import active_preparation
    declared = {node.node_id: node for node in spec.nodes}
    check_canceled()
    with TemporaryDirectory(prefix="bagelquant-core-kernel-") as directory:
        root, descriptors = Path(directory), []
        for index, (parent, value) in enumerate(scoped.items()):
            check_canceled()
            frame_file = f"input-{index}.arrow"
            value.lazy(include_traces=True).collect().write_ipc(root / frame_file)
            membership_file = None
            if value.domain.is_dynamic:
                membership_file = f"membership-{index}.arrow"
                value.domain.grid_lazy().collect().write_ipc(root / membership_file)
            descriptors.append(FiniteKernelInput(parent, declared[parent].input_key, frame_file,
                membership_file, tuple(value.domain.times), tuple(value.domain.asset_ids),
                value.domain.signature, value.identity, value.value_type.value, value.trace_columns))
        work = FiniteKernelWork(str(root), node_id, json.dumps(spec.to_dict()), json.dumps(parameters),
            tuple(descriptors), output_start, active_resource_limits(), estimated_payload_bytes, tuple(output_nodes))
        preparation = active_preparation()
        if preparation is not None:
            preparation.counts["dispatched_kernels"] += 1
            preparation.counts["transport_input_bytes"] += sum(path.stat().st_size for path in root.iterdir())
        result = dispatch(work)
        check_canceled()
        if not isinstance(result, FiniteKernelResult) or result.node_id != node_id or result.frame_file != "result.arrow":
            raise ValueError("finite worker returned a different result")
        path = _staged(root, result.frame_file)
        frame = pl.read_ipc(path, memory_map=False)
        if frame.height != result.rows:
            raise ValueError("finite worker row count differs")
        frames = {node_id: (frame, result.value_type, result.trace_columns)}
        total = path.stat().st_size
        for node, filename, kind, traces, rows in result.intermediates:
            if node in frames or node not in output_nodes:
                raise ValueError("finite worker returned an unrelated intermediate")
            selected = _staged(root, filename)
            values = pl.read_ipc(selected, memory_map=False)
            if values.height != rows:
                raise ValueError("finite worker intermediate row count differs")
            frames[node] = (values, kind, traces)
            total += selected.stat().st_size
        if set(frames) != set(output_nodes):
            raise ValueError("finite worker omitted an intermediate")
        from .graph import Graph
        from .graph_management import parents
        bound = Graph.from_logical_spec(LogicalGraphSpec({node: node for node in output_nodes}, spec.nodes),
            inputs={declared[parent].input_key: value for parent, value in scoped.items()
                if declared[parent].node_type == "input"},
            node_bindings={parent: value for parent, value in scoped.items() if declared[parent].node_type == "operator"},
            parameter_bindings=parameters)
        kinds = {node.logical_id: node.value_type.value for node in bound.nodes}
        trace_types = {node: value.trace_columns for node, value in scoped.items()}
        for node in spec.nodes:
            if node.node_id not in trace_types and node.node_type == "operator":
                trace_types[node.node_id] = tuple(dict.fromkeys(trace for parent in parents(node)
                    for trace in trace_types.get(parent, ())))
        for node, (values, kind, traces) in frames.items():
            if (kind != kinds[node] or tuple(traces) != trace_types[node]
                    or set(values.columns) != {"time", "asset_id", "value", *traces}):
                raise ValueError("finite worker changed its frozen value/trace type")
        if preparation is not None:
            preparation.counts["transport_output_bytes"] += total
        return frames
