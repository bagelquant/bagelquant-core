"""Trading-calendar and universe definitions for panel inputs."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ..hashing import hash_dataframe, hash_mapping

REGION_CALENDARS = {
    "US": "XNYS",
    "CN": "XSHG",
    "HK": "XHKG",
}
CALENDAR_CACHE_ENV = "BAGELQUANT_CALENDAR_CACHE_DIR"
CALENDAR_CACHE_VERSION = 1


def _normalize_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(pd.to_datetime(index))
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    return normalized.normalize().as_unit("ns")


def _calendar_cache_root() -> Path:
    override = os.environ.get(CALENDAR_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "bagelquant" / "calendars"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "bagelquant" / "calendars"
    return Path.home() / ".cache" / "bagelquant" / "calendars"


def _calendar_cache_path(calendar_name: str) -> Path:
    return _calendar_cache_root() / f"{calendar_name}.json"


def _sessions_from_cache(
    calendar_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex | None:
    try:
        payload = json.loads(_calendar_cache_path(calendar_name).read_text())
        if payload["version"] != CALENDAR_CACHE_VERSION:
            return None
        if payload["calendar"] != calendar_name:
            return None
        cached_start = _normalize_timestamp(payload["start_date"], name="start_date")
        cached_end = _normalize_timestamp(payload["end_date"], name="end_date")
        if start < cached_start or end > cached_end:
            return None
        sessions = _normalize_index(pd.Index(payload["sessions"]))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return sessions[(sessions >= start) & (sessions <= end)]


def _write_calendar_cache(
    calendar_name: str,
    sessions: pd.DatetimeIndex,
) -> None:
    if sessions.empty:
        return
    cache_path = _calendar_cache_path(calendar_name)
    temporary_path = cache_path.with_suffix(".tmp")
    payload = {
        "version": CALENDAR_CACHE_VERSION,
        "calendar": calendar_name,
        "start_date": sessions[0].isoformat(),
        "end_date": sessions[-1].isoformat(),
        "sessions": [session.isoformat() for session in sessions],
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(json.dumps(payload), encoding="utf-8")
        temporary_path.replace(cache_path)
    except OSError:
        return


def _retrieve_calendar_sessions(
    calendar_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar(calendar_name)
    sessions = _normalize_index(calendar.sessions)
    if sessions.empty or start < sessions[0] or end > sessions[-1]:
        calendar = xcals.get_calendar(
            calendar_name,
            start=min(start, sessions[0]) if not sessions.empty else start,
            end=max(end, sessions[-1]) if not sessions.empty else end,
        )
        sessions = _normalize_index(calendar.sessions)
    _write_calendar_cache(calendar_name, sessions)
    return sessions[(sessions >= start) & (sessions <= end)]


def _sessions_in_range(
    calendar_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    cached = _sessions_from_cache(calendar_name, start, end)
    if cached is not None:
        return cached
    return _retrieve_calendar_sessions(calendar_name, start, end)


class Domain:
    """Daily trading sessions and asset membership for panel data."""

    def __init__(
        self,
        *,
        region: str,
        universe: Sequence[Any] | pd.DataFrame,
        start_date: Any,
        end_date: Any,
    ) -> None:
        normalized_region = region.upper()
        if normalized_region not in REGION_CALENDARS:
            supported = ", ".join(sorted(REGION_CALENDARS))
            raise ValueError(f"Unsupported region '{region}'. Expected one of: {supported}")

        start = _normalize_timestamp(start_date, name="start_date")
        end = _normalize_timestamp(end_date, name="end_date")
        if start > end:
            raise ValueError("start_date must not be after end_date")

        calendar_name = REGION_CALENDARS[normalized_region]
        sessions = _sessions_in_range(calendar_name, start, end)

        self.region = normalized_region
        self.calendar_name = calendar_name
        self.start_date = start
        self.end_date = end
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
                "region": self.region,
                "calendar": self.calendar_name,
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
