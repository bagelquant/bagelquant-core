"""
Panel Module

Key concept in bagelquant-core, see docs/01_concepts/panel.md for more details.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .hashing import hash_dataframe
from .node import Node


class Panel(Node):
    """
    Immutable data container and leaf node in the DAG.

    Panels are the explicit data boundary of BagelQuant. Users create Panels
    from DataFrames and graph execution also produces Panels.
    """

    node_type = "panel"

    def __init__(
        self,
        data: pd.DataFrame,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, metadata=metadata)
        self._data = self._validate_data(data)
        self._data_hash = hash_dataframe(self._data)

    @property
    def parents(self) -> tuple[Node, ...]:
        return ()

    def compute(self) -> pd.DataFrame:
        return self._data

    @property
    def output(self) -> "Panel":
        return self

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def config(self) -> Mapping[str, Any]:
        return {"data_hash": self._data_hash}

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

        return tuple(frame.reindex(index=index, columns=columns) for frame in frames)

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
