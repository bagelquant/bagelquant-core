"""Immutable, lazily materialized domain-aware panel containers."""

from __future__ import annotations

from itertools import count
from typing import Any, Mapping, Sequence

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, align_frames, normalize_panel_frame
from ..node import Node
from .domain import Domain

_INTERNAL_MATERIALIZATION_TOKEN = object()
_IDENTITIES = count(1)


class Panel(Node):
    """Immutable numeric panel backed by a sparse Polars lazy plan."""

    node_type = "panel"

    def __init__(
        self,
        data: pl.LazyFrame,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        _domain: Domain | None = None,
        _token: object | None = None,
        _identity: str | None = None,
        _trace_identity: str | None = None,
        _trace_columns: Sequence[str] = (),
        _cached_dense: pl.DataFrame | None = None,
    ) -> None:
        if _token is not _INTERNAL_MATERIALIZATION_TOKEN or _domain is None:
            raise TypeError(
                f"Create panel inputs with {self.__class__.__name__}."
                "from_domain(data, domain)"
            )
        super().__init__(name=name, metadata=metadata)
        self._domain = _domain
        self._frame = data
        self._identity = _identity or f"panel:{next(_IDENTITIES)}"
        self._trace_columns = tuple(dict.fromkeys(_trace_columns))
        self._trace_identity = (
            _trace_identity or f"trace:{self._identity}"
            if self._trace_columns
            else None
        )
        self._cached_dense = _cached_dense
        # Compatibility for code that previously used the payload hash.
        self._data_hash = self._identity

    @classmethod
    def from_domain(
        cls,
        data: pl.DataFrame | pl.LazyFrame,
        domain: Domain,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        identity: str | None = None,
        trace_identity: str | None = None,
        trace_columns: Sequence[str] = (),
    ) -> "Panel":
        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain")
        traces = tuple(dict.fromkeys(str(value) for value in trace_columns))
        frame = cls._normalize_source(data, traces)
        frame = domain.apply_membership_lazy(frame)
        return cls(
            frame,
            name=name,
            metadata=metadata,
            _domain=domain,
            _token=_INTERNAL_MATERIALIZATION_TOKEN,
            _identity=identity,
            _trace_identity=trace_identity,
            _trace_columns=traces,
        )

    @classmethod
    def _from_plan(
        cls,
        frame: pl.LazyFrame,
        *,
        domain: Domain,
        name: str,
        metadata: Mapping[str, Any] | None,
        identity: str,
        trace_identity: str | None = None,
        trace_columns: Sequence[str],
        dense_output: bool,
    ) -> "Panel":
        panel = cls(
            frame,
            name=name,
            metadata=metadata,
            _domain=domain,
            _token=_INTERNAL_MATERIALIZATION_TOKEN,
            _identity=identity,
            _trace_identity=trace_identity,
            _trace_columns=trace_columns,
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
    ) -> "Panel":
        panel = cls.from_domain(data, domain, name=name, metadata=metadata)
        panel._cached_dense = panel._collect_and_validate(dense=True)
        return panel

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def trace_columns(self) -> tuple[str, ...]:
        return self._trace_columns

    @property
    def trace_identity(self) -> str | None:
        return self._trace_identity

    @property
    def parents(self) -> tuple[Node, ...]:
        return ()

    def compute(self, *inputs: pl.DataFrame) -> pl.DataFrame:
        if inputs:
            raise ValueError("Panel does not accept inputs")
        return self.data

    @property
    def output(self) -> "Panel":
        return self

    @property
    def data(self) -> pl.DataFrame:
        return self.collect(dense=True, include_traces=False)

    def collect(
        self,
        *,
        dense: bool = True,
        include_traces: bool = False,
    ) -> pl.DataFrame:
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

        frame = (
            self._domain.align_lazy(
                self._frame, trace_columns=self._trace_columns
            )
            if dense
            else self._domain.apply_membership_lazy(self._frame)
        )
        columns = [TIME, ASSET_ID, VALUE]
        if include_traces:
            columns.extend(self._trace_columns)
        return frame.select(columns)

    def config(self) -> Mapping[str, Any]:
        return {
            "input_identity": self._identity,
            "domain_signature": self._domain.signature,
            "trace_columns": self._trace_columns,
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
    ) -> pl.LazyFrame:
        required = {TIME, ASSET_ID, VALUE, *trace_columns}
        if isinstance(data, pl.DataFrame):
            missing = sorted(required - set(data.columns))
            if missing:
                raise ValueError(
                    f"panel data is missing required columns: {missing}"
                )
            base = cls._validate_data(data.select(TIME, ASSET_ID, VALUE))
            if not trace_columns:
                return base.lazy()
            traces = data.select(TIME, ASSET_ID, *trace_columns).with_columns(
                pl.col(TIME).cast(pl.Date, strict=False),
                pl.col(ASSET_ID).cast(pl.String),
            )
            return base.join(traces, on=[TIME, ASSET_ID], how="left").lazy()

        schema = data.collect_schema()
        missing = sorted(required - set(schema.names()))
        if missing:
            raise ValueError(f"panel data is missing required columns: {missing}")
        if cls is Panel and not schema[VALUE].is_numeric():
            raise TypeError("panel value column must be numeric")
        frame = data.select(TIME, ASSET_ID, VALUE, *trace_columns).with_columns(
            pl.col(TIME).cast(pl.Date, strict=False),
            pl.col(ASSET_ID).cast(pl.String),
        )
        if cls is Panel and schema[VALUE].is_float():
            frame = frame.with_columns(pl.col(VALUE).fill_nan(None))
        return frame.sort([TIME, ASSET_ID])

    def _collect_and_validate(self, *, dense: bool) -> pl.DataFrame:
        frame = (
            self._domain.align_lazy(
                self._frame, trace_columns=self._trace_columns
            )
            if dense
            else self._domain.apply_membership_lazy(self._frame)
        )
        return self._validate_collected(frame.collect())

    def _validate_collected(self, collected: pl.DataFrame) -> pl.DataFrame:
        base = self._validate_data(collected.select(TIME, ASSET_ID, VALUE))
        if not self._trace_columns:
            return base
        traces = collected.select(
            TIME, ASSET_ID, *self._trace_columns
        ).with_columns(
            pl.col(TIME).cast(pl.Date, strict=False),
            pl.col(ASSET_ID).cast(pl.String),
        )
        return base.join(traces, on=[TIME, ASSET_ID], how="left").sort(
            [TIME, ASSET_ID]
        )

    @staticmethod
    def _validate_data(data: pl.DataFrame) -> pl.DataFrame:
        return normalize_panel_frame(data, numeric=True)


class CategoryPanel(Panel):
    """Immutable categorical domain-aware data container."""

    @staticmethod
    def _validate_data(data: pl.DataFrame) -> pl.DataFrame:
        return normalize_panel_frame(data, numeric=False)
