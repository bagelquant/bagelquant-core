"""Immutable, lazily materialized domain-aware panel containers."""

from __future__ import annotations

from uuid import uuid4
from .hashing import hash_dataframe
from typing import Any, Mapping, Sequence

import polars as pl

from .frame import (
    ASSET_ID,
    TIME,
    VALUE,
    align_frames,
    normalize_date_expression,
    normalize_panel_frame,
)
from ._node_definition import NodeDefinition
from .domain import Domain
from enum import StrEnum

_INTERNAL_MATERIALIZATION_TOKEN = object()


class ValueType(StrEnum):
    NUMERIC = "numeric"
    CATEGORY = "category"
    PREDICTION = "prediction"
    WEIGHTS = "weights"


class Node(NodeDefinition):
    """Immutable numeric panel backed by a sparse Polars lazy plan."""

    node_type = "input"

    def __init__(
        self,
        data: pl.LazyFrame,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        value_type: ValueType | str = ValueType.NUMERIC,
        _domain: Domain | None = None,
        _token: object | None = None,
        _identity: str | None = None,
        _trace_identity: str | None = None,
        _trace_columns: Sequence[str] = (),
        _cached_dense: pl.DataFrame | None = None,
        _validated_keys: bool = False,
        _exact_domain: bool = False,
        _key_identity: str | None = None,
        _source_key: str | None = None,
    ) -> None:
        if _token is not _INTERNAL_MATERIALIZATION_TOKEN or _domain is None:
            raise TypeError(
                f"Create panel inputs with {self.__class__.__name__}."
                "from_domain(data, domain)"
            )
        super().__init__(name=name, metadata=metadata)
        self._value_type = ValueType(value_type)
        self._domain = _domain
        self._frame = data
        self._identity = _identity or f"transient:{uuid4().hex}"
        self._durable_identity = _identity is not None
        self._trace_columns = tuple(dict.fromkeys(_trace_columns))
        self._trace_identity = (
            _trace_identity or f"trace:{self._identity}"
            if self._trace_columns
            else None
        )
        self._cached_dense = _cached_dense
        self._validated_keys = _validated_keys
        self._exact_domain = _exact_domain
        self._key_identity = _key_identity or self._identity
        self._source_key = _source_key or f"input:{uuid4().hex}"

    @classmethod
    def from_domain(
        cls,
        data: pl.DataFrame | pl.LazyFrame,
        domain: Domain,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        value_type: ValueType | str = ValueType.NUMERIC,
        identity: str | None = None,
        trace_identity: str | None = None,
        trace_columns: Sequence[str] = (),
        source_key: str | None = None,
    ) -> "Node":
        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain")
        traces = tuple(dict.fromkeys(str(value) for value in trace_columns))
        validated_keys = isinstance(data, pl.DataFrame)
        frame = cls._normalize_source(data, traces, value_type=ValueType(value_type))
        exact_domain = (
            isinstance(data, pl.DataFrame)
            and domain._contains_exact_keys(data)
        )
        frame = domain.apply_membership_lazy(frame)
        if identity is None and isinstance(data, pl.DataFrame):
            identity = "content:" + hash_dataframe(frame.sort([TIME, ASSET_ID]).collect())
        return cls(
            frame,
            value_type=value_type,
            name=name,
            metadata=metadata,
            _domain=domain,
            _token=_INTERNAL_MATERIALIZATION_TOKEN,
            _identity=identity,
            _trace_identity=trace_identity,
            _trace_columns=traces,
            _validated_keys=validated_keys,
            _exact_domain=exact_domain,
            _key_identity=(
                f"domain:{domain.signature}"
                if exact_domain
                else None
            ),
            _source_key=source_key,
        )

    @classmethod
    def _from_plan(
        cls,
        frame: pl.LazyFrame,
        *,
        value_type: ValueType | str = ValueType.NUMERIC,
        domain: Domain,
        name: str,
        metadata: Mapping[str, Any] | None,
        identity: str,
        trace_identity: str | None = None,
        trace_columns: Sequence[str],
        dense_output: bool,
        validated_keys: bool = False,
        exact_domain: bool = False,
        key_identity: str | None = None,
    ) -> "Node":
        panel = cls(
            frame,
            value_type=value_type,
            name=name,
            metadata=metadata,
            _domain=domain,
            _token=_INTERNAL_MATERIALIZATION_TOKEN,
            _identity=identity,
            _trace_identity=trace_identity,
            _trace_columns=trace_columns,
            _validated_keys=validated_keys,
            _exact_domain=exact_domain,
            _key_identity=key_identity,
        )
        if dense_output:
            panel._cached_dense = panel._collect_and_validate(dense=True)
        return panel

    @classmethod
    def _materialize(
        cls,
        data: pl.DataFrame,
        *,
        domain: Domain,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Node":
        panel = cls.from_domain(data, domain, name=name, metadata=metadata)
        panel._cached_dense = panel._collect_and_validate(dense=True)
        return panel

    @property
    def value_type(self) -> ValueType:
        return self._value_type

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def source_key(self) -> str:
        """Permanent symbolic input key, independent of payload and Domain."""

        return self._source_key

    @property
    def trace_columns(self) -> tuple[str, ...]:
        return self._trace_columns

    @property
    def trace_identity(self) -> str | None:
        return self._trace_identity

    @property
    def parents(self) -> tuple[Node, ...]:
        return getattr(self, "_parents", ())

    def compute(self, *, runtime=None, dense_output: bool = True):
        """Explicitly evaluate this dependency closure and bind its result."""
        from .graph import Graph
        return Graph._from_nodes((self,)).compute(runtime=runtime, dense_output=dense_output)

    @property
    def output(self) -> "Node":
        if self.node_type == "input":
            return self
        return super().output

    def collect(
        self,
        *,
        dense: bool = True,
        include_traces: bool = False,
    ) -> pl.DataFrame:
        if self.node_type == "operator":
            return self.output.collect(dense=dense, include_traces=include_traces)
        if self._frame is None:
            raise RuntimeError(f"source {self.source_key!r} requires an explicit data binding")
        if dense and self._cached_dense is not None:
            result = self._cached_dense
        else:
            result = self._collect_and_validate(dense=dense)
            if dense:
                self._cached_dense = result
        columns = [TIME, ASSET_ID, VALUE]
        if include_traces:
            columns.extend(self._trace_columns)
        return result.select(columns).clone()

    def lazy(
        self,
        *,
        dense: bool = False,
        include_traces: bool = False,
    ) -> pl.LazyFrame:
        """Return a read-only lazy view for downstream plan composition."""

        if self.node_type == "operator":
            return self.output.lazy(dense=dense, include_traces=include_traces)
        if self._frame is None:
            raise RuntimeError(f"source {self.source_key!r} requires an explicit data binding")
        frame = (
            self._domain.align_lazy(
                self._frame, trace_columns=self._trace_columns
            )
            if dense
            else self._frame
        )
        columns = [TIME, ASSET_ID, VALUE]
        if include_traces:
            columns.extend(self._trace_columns)
        return frame.select(columns)

    def config(self) -> Mapping[str, Any]:
        if self.node_type == "operator":
            return {"operator": self._registered.registry_name, **self._config}
        return {
            "input_identity": self._identity,
            "domain_signature": None if self._domain is None else self._domain.signature,
            "trace_columns": self._trace_columns,
            "value_type": self.value_type.value,
        }

    @staticmethod
    def align_frames(
        *frames: pl.DataFrame, join: str = "inner"
    ) -> tuple[pl.DataFrame, ...]:
        return align_frames(*frames, join=join)

    @classmethod
    def _normalize_source(
        cls,
        data: pl.DataFrame | pl.LazyFrame,
        trace_columns: Sequence[str],
        *, value_type: ValueType = ValueType.NUMERIC,
    ) -> pl.LazyFrame:
        required = {TIME, ASSET_ID, VALUE, *trace_columns}
        if isinstance(data, pl.DataFrame):
            missing = sorted(required - set(data.columns))
            if missing:
                raise ValueError(
                    f"panel data is missing required columns: {missing}"
                )
            base = normalize_panel_frame(data.select(TIME, ASSET_ID, VALUE), numeric=value_type != ValueType.CATEGORY)
            if not trace_columns:
                return base.lazy()
            traces = data.select(TIME, ASSET_ID, *trace_columns).with_columns(
                normalize_date_expression(TIME, data.schema[TIME]),
                pl.col(ASSET_ID).cast(pl.String),
            )
            return base.join(
                traces,
                on=[TIME, ASSET_ID],
                how="left",
                maintain_order="left",
            ).lazy()

        schema = data.collect_schema()
        missing = sorted(required - set(schema.names()))
        if missing:
            raise ValueError(f"panel data is missing required columns: {missing}")
        if value_type != ValueType.CATEGORY and not schema[VALUE].is_numeric():
            raise TypeError("panel value column must be numeric")
        frame = data.select(TIME, ASSET_ID, VALUE, *trace_columns).with_columns(
            normalize_date_expression(TIME, schema[TIME]),
            pl.col(ASSET_ID).cast(pl.String),
        )
        if value_type != ValueType.CATEGORY and schema[VALUE].is_float():
            frame = frame.with_columns(pl.col(VALUE).fill_nan(None))
        return frame.sort([TIME, ASSET_ID])

    def _collect_and_validate(self, *, dense: bool) -> pl.DataFrame:
        if self._frame is None:
            raise RuntimeError(f"source {self.source_key!r} requires an explicit data binding")
        frame = (
            self._domain.align_lazy(
                self._frame, trace_columns=self._trace_columns
            )
            if dense
            else self._frame
        )
        return self._validate_collected(frame.collect())

    def _validate_collected(self, collected: pl.DataFrame) -> pl.DataFrame:
        base = self._validate_data(collected.select(TIME, ASSET_ID, VALUE))
        if not self._trace_columns:
            return base
        traces = collected.select(
            TIME, ASSET_ID, *self._trace_columns
        ).with_columns(
            normalize_date_expression(TIME, collected.schema[TIME]),
            pl.col(ASSET_ID).cast(pl.String),
        )
        return base.join(traces, on=[TIME, ASSET_ID], how="left").sort(
            [TIME, ASSET_ID]
        )

    def _validate_data(self, data: pl.DataFrame) -> pl.DataFrame:
        return normalize_panel_frame(data, numeric=self.value_type != ValueType.CATEGORY)


    @classmethod
    def from_operation(cls, *, inputs, panel_parameters, operation, config, name,
                       metadata=None):
        """Construct one deferred typed result with all dependencies explicit."""
        from .operator import OPERATOR_REGISTRY
        registered = operation if hasattr(operation, "registry_name") else OPERATOR_REGISTRY.get("prediction:" + operation.kind)
        if registered.registry_name == "bagelquant_core.operator.basic.identity" and not config and not panel_parameters:
            from copy import copy
            registered.resolve_type(inputs, panel_parameters)
            for check in getattr(inputs[0], "_execution_guards", ()):
                check()
            node = copy(inputs[0])
            node.name = name
            node.metadata = dict(metadata or {})
            return node
        node = cls.__new__(cls)
        NodeDefinition.__init__(node, name=name, metadata=metadata)
        node.node_type = "operator"
        if registered.registry_name in {"bagelquant_core.operator.arithmetic.add", "bagelquant_core.operator.arithmetic.mul"}:
            inputs = tuple(sorted(inputs, key=lambda value: value.logical_id))
        node._inputs = tuple(inputs)
        node._panel_parameters = dict(sorted(panel_parameters.items()))
        node._parents = (*node._inputs, *(n for values in node._panel_parameters.values() for n in values))
        node._execution_guards = tuple(dict.fromkeys(guard for parent in node._parents for guard in getattr(parent, "_execution_guards", ())))
        for guard in node._execution_guards:
            guard()
        node._operation = operation
        node._registered = registered
        node._config = dict(config)
        node._value_type = registered.resolve_type(node._inputs, node._panel_parameters)
        if registered.factory is None:
            from inspect import signature
            signature(registered.operation).bind(*node._inputs,
                **{role: values if registered.panel_parameter_kinds[role] else values[0] for role, values in node._panel_parameters.items()}, **node._config)
        node._domain = node._parents[-1].domain if registered.registry_name.endswith(".project_domain") else node._parents[0].domain
        return node

    @property
    def operation(self):
        return self._operation

    @property
    def contract(self):
        return self._registered.contract

    def spec_inputs(self):
        return self._inputs if self.node_type == "operator" else ()

    def spec_panel_parameters(self):
        return self._panel_parameters if self.node_type == "operator" else {}

    def evaluate_frames(self, *frames):
        if len(frames) != len(self.parents):
            raise ValueError(f"{self.name} requires {len(self.parents)} dependency frames")
        auxiliary = {}
        offset = len(self._inputs)
        for parameter, nodes in self._panel_parameters.items():
            values = frames[offset:offset + len(nodes)]
            offset += len(nodes)
            auxiliary[parameter] = tuple(values) if self._registered.panel_parameter_kinds[parameter] else values[0]
        if self._registered.factory is not None:
            return self.operation._compute_frames(*frames[:len(self._inputs)], **auxiliary)
        from .operator_state import operator_checkpoint_node
        from .logical import normalized_operator_config
        signature = {"operator": self._registered.registry_name, "version": self._registered.version,
                     "parameters": normalized_operator_config(self._registered.registry_name, self._config)}
        with operator_checkpoint_node(self.logical_id, signature):
            return self._registered.operation(*frames[:len(self._inputs)], **auxiliary, **self._config)

    def spec(self):
        """Export this value's complete dependency graph."""
        return self.graph.spec()

    def definition(self):
        return super().spec()

    @property
    def graph(self):
        from .graph import Graph
        return Graph(outputs=(self,))

    def logical_spec(self, *, input_keys=None):
        return self.graph.logical_spec(input_keys=input_keys)

    @classmethod
    def symbolic(cls, source_key, *, name=None, value_type="numeric"):
        """Declare a source without reading a payload; bind it before execution."""
        node = cls.__new__(cls)
        NodeDefinition.__init__(node, name=name or source_key)
        node._value_type = ValueType(value_type)
        node._source_key = source_key
        node._domain = None
        node._frame = None
        node._identity = None
        node._trace_columns = ()
        node._trace_identity = None
        node._cached_dense = None
        return node
