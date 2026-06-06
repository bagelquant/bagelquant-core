"""Explicit trading-calendar and universe definitions for panel inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..hashing import hash_dataframe, hash_mapping


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(pd.to_datetime(index))
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    return normalized.normalize().as_unit("ns")


def _normalize_calendar(calendar: Sequence[Any] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    sessions = _normalize_index(pd.Index(calendar))
    if sessions.empty:
        raise ValueError("calendar must contain at least one session")
    if sessions.has_duplicates:
        raise ValueError("calendar sessions must be unique")
    if sessions.hasnans:
        raise ValueError("calendar sessions must be valid dates")
    if not sessions.is_monotonic_increasing:
        raise ValueError("calendar sessions must be sorted ascending")
    return sessions


class Domain:
    """Daily trading sessions and asset membership for panel data."""

    def __init__(
        self,
        *,
        calendar: Sequence[Any] | pd.DatetimeIndex,
        universe: Sequence[Any] | pd.DataFrame,
    ) -> None:
        sessions = _normalize_calendar(calendar)

        self.start_date = sessions[0]
        self.end_date = sessions[-1]
        self._sessions = sessions

        if isinstance(universe, pd.DataFrame):
            self._is_dynamic = True
            self._membership = self._normalize_dynamic_membership(universe)
        else:
            self._is_dynamic = False
            self._membership = self._static_membership(universe)

        self._assets = self._membership.columns.copy()
        self._signature = hash_mapping(
            {
                "calendar": hash_dataframe(
                    pd.DataFrame(index=self._sessions).assign(session=True)
                ),
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "membership": hash_dataframe(self._membership),
            }
        )

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self._sessions.copy()

    @property
    def assets(self) -> pd.Index:
        return self._assets.copy()

    @property
    def membership(self) -> pd.DataFrame:
        return self._membership.copy(deep=True)

    @property
    def is_dynamic(self) -> bool:
        return self._is_dynamic

    @property
    def signature(self) -> str:
        return self._signature

    def empty_frame(self) -> pd.DataFrame:
        """Return a NaN-filled frame with this Domain's exact dimensions."""

        return pd.DataFrame(np.nan, index=self._sessions, columns=self._assets)

    def normalize_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align input data to sessions and assets, masking inactive cells."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Panel data must be a pandas DataFrame")
        if data.index.nlevels != 1 or data.columns.nlevels != 1:
            raise ValueError("Panel data must have 1D index and columns")
        if data.index.has_duplicates or data.columns.has_duplicates:
            raise ValueError("Panel index and columns must be unique")

        normalized = data.copy(deep=True)
        normalized.index = _normalize_index(normalized.index)
        if normalized.index.has_duplicates:
            raise ValueError("Panel index must remain unique after date normalization")
        normalized = normalized.reindex(index=self._sessions, columns=self._assets)
        return self.apply_membership(normalized)

    def apply_membership(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mask inactive dynamic-universe cells while avoiding static overhead."""

        if not self._is_dynamic:
            return data
        return data.where(self._membership)

    def equivalent_to(self, other: object) -> bool:
        return isinstance(other, Domain) and self.signature == other.signature

    def _static_membership(self, universe: Sequence[Any]) -> pd.DataFrame:
        if isinstance(universe, (str, bytes)):
            raise TypeError("Static universe must be a sequence of asset labels")
        assets = pd.Index(universe)
        self._validate_assets(assets)
        return pd.DataFrame(True, index=self._sessions, columns=assets)

    def _normalize_dynamic_membership(self, universe: pd.DataFrame) -> pd.DataFrame:
        if universe.index.nlevels != 1 or universe.columns.nlevels != 1:
            raise ValueError("Dynamic universe must have 1D index and columns")
        if universe.index.has_duplicates or universe.columns.has_duplicates:
            raise ValueError("Dynamic universe index and columns must be unique")
        self._validate_assets(universe.columns)

        membership = universe.copy(deep=True)
        membership.index = _normalize_index(membership.index)
        if membership.index.has_duplicates:
            raise ValueError(
                "Dynamic universe index must remain unique after date normalization"
            )

        provided = membership.stack(future_stack=True).dropna()
        if not provided.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise TypeError("Dynamic universe values must be boolean or missing")

        return membership.reindex(index=self._sessions, columns=membership.columns).fillna(
            False
        ).astype(bool)

    @staticmethod
    def _validate_assets(assets: pd.Index) -> None:
        if assets.empty:
            raise ValueError("Universe must contain at least one asset")
        if assets.has_duplicates:
            raise ValueError("Universe assets must be unique")
