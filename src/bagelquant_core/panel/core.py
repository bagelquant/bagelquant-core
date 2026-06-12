"""Immutable domain-aware panel data containers."""

from __future__ import annotations

from typing import Any, Mapping

import polars as pl

from ..frame import align_frames, normalize_panel_frame
from ..hashing import hash_dataframe
from ..node import Node
from .domain import Domain

_INTERNAL_MATERIALIZATION_TOKEN = object()


class Panel(Node):
    """Immutable numeric data container and leaf node in the DAG."""

    node_type = "panel"

    def __init__(
        self,
        data: pl.DataFrame,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        _domain: Domain | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _INTERNAL_MATERIALIZATION_TOKEN or _domain is None:
            raise TypeError(
                f"Create panel inputs with {self.__class__.__name__}.from_domain(data, domain)"
            )
        super().__init__(name=name, metadata=metadata)
        self._domain = _domain
        self._data = self._validate_data(data)
        self._data_hash = hash_dataframe(self._data)

    @classmethod
    def from_domain(
        cls,
        data: pl.DataFrame,
        domain: Domain,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Panel":
        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain")
        return cls._materialize(data, domain=domain, name=name, metadata=metadata)

    @classmethod
    def _materialize(
        cls,
        data: pl.DataFrame,
        *,
        domain: Domain,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Panel":
        normalized = domain.normalize_frame(data)
        return cls(
            normalized,
            name=name,
            metadata=metadata,
            _domain=domain,
            _token=_INTERNAL_MATERIALIZATION_TOKEN,
        )

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def parents(self) -> tuple[Node, ...]:
        return ()

    def compute(self, *inputs: pl.DataFrame) -> pl.DataFrame:
        if inputs:
            raise ValueError("Panel does not accept inputs")
        return self._data.clone()

    @property
    def output(self) -> "Panel":
        return self

    @property
    def data(self) -> pl.DataFrame:
        return self._data.clone()

    def config(self) -> Mapping[str, Any]:
        return {
            "data_hash": self._data_hash,
            "domain_signature": self._domain.signature,
        }

    @staticmethod
    def align_frames(
        *frames: pl.DataFrame, join: str = "inner"
    ) -> tuple[pl.DataFrame, ...]:
        return align_frames(*frames, join=join)

    @staticmethod
    def _validate_data(data: pl.DataFrame) -> pl.DataFrame:
        return normalize_panel_frame(data, numeric=True)


class CategoryPanel(Panel):
    """Immutable categorical domain-aware data container and leaf node."""

    @staticmethod
    def _validate_data(data: pl.DataFrame) -> pl.DataFrame:
        return normalize_panel_frame(data, numeric=False)
