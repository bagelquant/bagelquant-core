"""Operation-local coordinate queries charged to an admitted finite worker."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import TracebackType
from typing import Self, cast

import numpy as np
from numpy.typing import NDArray
import polars as pl

from .domain import Domain
from .resources import ResourceLimits


_MIB = 1024 * 1024
_OBJECT_ALLOWANCE = 4096
_EPOCH = date(1970, 1, 1)


class _FiniteTimeIndex:
    """Owned dates grouped by anonymous asset offsets; never retain values."""

    def __init__(self) -> None:
        self.times: NDArray[np.int32] | None = None
        self.offsets: NDArray[np.int64] | None = None
        self.calendar: NDArray[np.int32] | None = None

    @property
    def retained_bytes(self) -> int:
        return sum(value.nbytes for value in (self.times, self.offsets, self.calendar)
                   if value is not None)

    def build(self, membership: pl.DataFrame, calendar: NDArray[np.int32]) -> None:
        """Copy only final buffers from the existing asset/time sorted frame."""
        lengths = value = None
        try:
            self.times = np.array(membership["time"].cast(pl.Int32).to_numpy(),
                                  dtype=np.int32, copy=True)
            lengths = membership["asset_id"].rle().struct.field("len").to_numpy()
            self.offsets = np.empty(len(lengths) + 1, dtype=np.int64)
            self.offsets[0] = 0
            lengths.cumsum(dtype=np.int64, out=self.offsets[1:])
            self.calendar = np.array(calendar, dtype=np.int32, copy=True)
            if int(self.offsets[-1]) != len(self.times):
                raise ValueError("finite membership offsets differ from admitted keys")
            for value in (self.times, self.offsets, self.calendar):
                value.flags.writeable = False
        finally:
            lengths = value = None
            del membership, calendar

    def close(self) -> None:
        self.times = self.offsets = self.calendar = None

    def _buffers(self) -> tuple[NDArray[np.int32], NDArray[np.int64], NDArray[np.int32]]:
        if self.times is None or self.offsets is None or self.calendar is None:
            raise ValueError("finite membership index is closed or incomplete")
        return self.times, self.offsets, self.calendar

    def canonical_halos(self, observations: int, sessions: int) -> dict[int, date]:
        times, offsets, calendar = self._buffers()
        origins = earliest = asset_times = counts = present = None
        try:
            origins = np.arange(0, len(calendar), sessions)
            earliest = calendar[np.maximum(0, origins - observations)].copy()
            if observations:
                for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
                    asset_times = times[int(begin):int(end)]
                    counts = asset_times.searchsorted(calendar[origins], side="left")
                    present = counts > 0
                    earliest[present] = np.minimum(earliest[present],
                        asset_times[np.maximum(0, counts[present] - observations)])
            earliest[0] = calendar[0]
            return {int(origin): _EPOCH + timedelta(days=int(day))
                    for origin, day in zip(origins, earliest, strict=True)}
        finally:
            times = offsets = calendar = origins = earliest = asset_times = counts = present = None

    def numerical_read_start(self, output_start: date, end: date, history: int) -> date:
        """Count admitted keys, including null-valued rows, for output assets."""
        if history == 0:
            return output_start
        times, offsets, calendar = self._buffers()
        asset_times = None
        try:
            first_day = (output_start - _EPOCH).days
            last_day = (end - _EPOCH).days
            position = int(calendar.searchsorted(first_day, side="left"))
            if position == len(calendar) or int(calendar[position]) != first_day:
                raise ValueError(f"{output_start!r} is not in list")
            earliest = int(calendar[max(0, position - history)])
            for begin, ending in zip(offsets[:-1], offsets[1:], strict=True):
                asset_times = times[int(begin):int(ending)]
                count = int(asset_times.searchsorted(first_day, side="left"))
                if 0 < count < len(asset_times) and int(asset_times[count]) <= last_day:
                    earliest = min(earliest, int(asset_times[max(0, count - history)]))
            return _EPOCH + timedelta(days=earliest)
        finally:
            times = offsets = calendar = asset_times = None


class _FiniteTimeIndexLease:
    """A private finite-call reservation, with no buffers before admission."""

    def __init__(self, domain: Domain, history: int, limits: ResourceLimits, *, eligible: bool,
                 canonical_sessions: int):
        self.limits = limits
        self.admitted = False
        self.closed = False
        self.raw_bytes = 0
        self.reserved_bytes = 0
        self.scratch_bytes = 0
        self.actual_retained_bytes = 0
        self.index: _FiniteTimeIndex | None = None
        self._domain: Domain | None = None
        if not eligible or history <= 0 or not domain.is_dynamic or domain.size == 0:
            return
        rows = domain.size
        assets = domain.asset_ids
        groups, sessions = len(assets), len(domain.times)
        max_asset_bytes = int(cast(int, assets.str.len_bytes().max() or 0))
        self.raw_bytes = 4 * rows + 8 * (groups + 1) + 4 * sessions + _OBJECT_ALLOWANCE
        reserved_mib = (self.raw_bytes + _MIB - 1) // _MIB
        # This is a conservative native-sort admission estimate, not an RSS cap.
        membership_upper = rows * (4 + 8 + max_asset_bytes)
        self.scratch_bytes = (4 * membership_upper + 8 * rows + 4 * sessions
            + 64 * ((sessions + canonical_sessions - 1) // canonical_sessions)
            + 8 * (groups + 1) + _OBJECT_ALLOWANCE)
        noncache_bytes = (limits.memory_target_mib - limits.cache_mib) * _MIB
        if reserved_mib > limits.cache_mib or self.scratch_bytes > noncache_bytes // 8:
            return
        self.reserved_bytes = reserved_mib * _MIB
        self.limits = replace(limits, cache_mib=limits.cache_mib - reserved_mib)
        self._domain = domain
        self.admitted = True

    def accepts(self, domain: Domain) -> bool:
        return self.admitted and not self.closed and self._domain is domain

    def __enter__(self) -> Self:
        if self.closed:
            raise ValueError("finite membership reservation is closed")
        return self

    def __exit__(self, _kind: type[BaseException] | None, _error: BaseException | None,
                 _traceback: TracebackType | None) -> None:
        self.close()

    def build_from_sorted(self, membership: pl.DataFrame,
                          calendar: NDArray[np.int32]) -> _FiniteTimeIndex:
        if not self.admitted or self.closed or self.index is not None:
            raise ValueError("finite membership reservation cannot build an index")
        self.index = _FiniteTimeIndex()
        try:
            self.index.build(membership, calendar)
            if self.index.retained_bytes + _OBJECT_ALLOWANCE > self.raw_bytes:
                raise ValueError("finite membership buffers exceed their admitted charge")
            self.actual_retained_bytes = self.index.retained_bytes
            return self.index
        except BaseException:
            self.index.close()
            self.index = None
            raise
        finally:
            del membership, calendar

    def close(self) -> None:
        if self.index is not None:
            self.index.close()
            self.index = None
        self._domain = None
        self.closed = True
