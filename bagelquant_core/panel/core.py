"""Immutable domain-aware panel data containers."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..hashing import hash_dataframe
from ..node import Node
from .domain import Domain

_INTERNAL_MATERIALIZATION_TOKEN = object()


class Panel(Node):
    """Immutable numeric data container and leaf node in the DAG."""

    node_type = "panel"

    def __init__(
        self,
        data: pd.DataFrame,
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
        data: pd.DataFrame,
        domain: Domain,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Panel":
        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain")
        return cls._materialize(
            data,
            domain=domain,
            name=name,
            metadata=metadata,
        )

    @classmethod
    def _materialize(
        cls,
        data: pd.DataFrame,
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

    def compute(self, *inputs: pd.DataFrame) -> pd.DataFrame:
        if inputs:
            raise ValueError("Panel does not accept inputs")
        return self._data.copy(deep=True)

    @property
    def output(self) -> "Panel":
        return self

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy(deep=True)

    def config(self) -> Mapping[str, Any]:
        return {
            "data_hash": self._data_hash,
            "domain_signature": self._domain.signature,
        }

    @staticmethod
    def align_frames(
        *frames: pd.DataFrame,
        join: str = "inner",
    ) -> tuple[pd.DataFrame, ...]:
        if join not in {"inner", "outer"}:
            raise ValueError("join must be either 'inner' or 'outer'")
        if len(frames) <= 1:
            return tuple(frames)

        index = frames[0].index
        columns = frames[0].columns
        for frame in frames[1:]:
            if join == "inner":
                index = index.intersection(frame.index)
                columns = columns.intersection(frame.columns)
            else:
                index = index.union(frame.index)
                columns = columns.union(frame.columns)

        return tuple(
            frame
            if frame.index.equals(index) and frame.columns.equals(columns)
            else frame.reindex(index=index, columns=columns)
            for frame in frames
        )

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Panel data must be a pandas DataFrame")
        if data.index.nlevels != 1 or data.columns.nlevels != 1:
            raise ValueError("Panel data must have 1D index and columns")
        if data.index.has_duplicates or data.columns.has_duplicates:
            raise ValueError("Panel index and columns must be unique")

        numeric_columns = data.select_dtypes(include="number").columns
        if len(numeric_columns) != len(data.columns):
            raise TypeError("Panel data must be fully numeric")

        validated = data.copy(deep=True)
        values = validated.to_numpy(copy=False)
        if hasattr(values, "flags"):
            values.flags.writeable = False
        return validated


class CategoryPanel(Panel):
    """Immutable categorical domain-aware data container and leaf node."""

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("CategoryPanel data must be a pandas DataFrame")
        if data.index.nlevels != 1 or data.columns.nlevels != 1:
            raise ValueError("CategoryPanel data must have 1D index and columns")
        if data.index.has_duplicates or data.columns.has_duplicates:
            raise ValueError("CategoryPanel index and columns must be unique")

        validated = data.copy(deep=True)
        values = validated.to_numpy(copy=False)
        if hasattr(values, "flags"):
            values.flags.writeable = False
        return validated
