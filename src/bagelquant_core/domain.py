"""Explicit trading-calendar and universe definitions for panel inputs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import polars as pl

from bagelquant_core.frame import (
    ASSET_ID,
    TIME,
    VALUE,
    normalize_asset_ids,
    normalize_date_expression,
    normalize_time_series,
)
from bagelquant_core.hashing import DATAFRAME_HASH_V1, hash_dataframe, hash_mapping


def _inclusive_date_count(values: Sequence[date] | pl.Series, start: date, end: date) -> int:
    """Count a sorted normalized Date axis using only scalar item lookups.

    Duplicate dates are intentional for active keys. No array conversion,
    expression evaluation, rechunk, index or retained cache is needed. Callers
    own the immutable sorted axis; this private helper does not normalize it.
    """
    item = None
    try:
        if type(start) is not date or type(end) is not date:
            raise TypeError("private date counts require datetime.date bounds")
        if start > end:
            return 0
        item = values.item if isinstance(values, pl.Series) else values.__getitem__
        lower, upper = 0, len(values)
        while lower < upper:
            middle = (lower + upper) // 2
            if item(middle) < start:
                lower = middle + 1
            else:
                upper = middle
        first, upper = lower, len(values)
        while lower < upper:
            middle = (lower + upper) // 2
            if item(middle) <= end:
                lower = middle + 1
            else:
                upper = middle
        return lower - first
    finally:
        # An error's traceback must not keep a new borrowed-Series alias alive.
        item = None
        del values


class Domain:
    """Trading times and compact static or dynamic asset membership.

    A static domain stores ``O(times + assets)`` state.  A dynamic domain stores
    only active keys.  The potentially large time-by-asset grid is a lazy plan
    and is collected only at a public dense boundary.
    """

    def __init__(
        self,
        *,
        calendar: Sequence[Any] | pl.Series,
        universe: Sequence[Any] | pl.Series | pl.DataFrame,
        identity: str | None = None,
    ) -> None:
        times = normalize_time_series(calendar)
        self.start_time = times[0]
        self.end_time = times[-1]
        self._times = times

        if isinstance(universe, pl.DataFrame):
            self._is_dynamic = True
            normalized = self._normalize_dynamic_membership(universe)
            self._active_membership = normalized.filter(pl.col("active")).select(
                TIME, ASSET_ID
            )
            active_assets = (
                self._active_membership.select(ASSET_ID)
                .unique()
                .sort(ASSET_ID)[ASSET_ID]
            )
            self._asset_ids = (
                pl.Series(ASSET_ID, [], dtype=pl.String)
                if active_assets.is_empty()
                else normalize_asset_ids(active_assets)
            )
        else:
            self._is_dynamic = False
            self._asset_ids = normalize_asset_ids(universe).sort()
            self._active_membership = None

        self._set_signature(identity)

    def _set_signature(self, identity: str | None = None) -> None:
        self._signature = identity or hash_mapping(
            {
                "calendar": hash_dataframe(pl.DataFrame({TIME: self._times}), policy=DATAFRAME_HASH_V1),
                "assets": hash_dataframe(
                    pl.DataFrame({ASSET_ID: self._asset_ids}), policy=DATAFRAME_HASH_V1
                ),
                "dynamic": self._is_dynamic,
                "active": (
                    None
                    if self._active_membership is None
                    else hash_dataframe(self._active_membership, policy=DATAFRAME_HASH_V1)
                ),
            }
        )

    @property
    def times(self) -> pl.Series:
        return self._times.clone()

    @property
    def asset_ids(self) -> pl.Series:
        return self._asset_ids.clone()

    @property
    def membership(self) -> pl.DataFrame:
        if not self._is_dynamic:
            return self._all_grid_lazy().with_columns(
                pl.lit(True).alias("active")
            ).collect()
        assert self._active_membership is not None
        return (
            self._all_grid_lazy()
            .join(
                self._active_membership.lazy().with_columns(
                    pl.lit(True).alias("active")
                ),
                on=[TIME, ASSET_ID],
                how="left",
            )
            .with_columns(pl.col("active").fill_null(False))
            .sort([TIME, ASSET_ID])
            .collect()
        )

    @property
    def is_dynamic(self) -> bool:
        return self._is_dynamic

    @property
    def signature(self) -> str:
        return self._signature

    @property
    def size(self) -> int:
        if self._is_dynamic:
            assert self._active_membership is not None
            return self._active_membership.height
        return len(self._times) * len(self._asset_ids)

    def _count_rows(self, start: date, end: date) -> int:
        """Count inclusive grid keys without constructing a window or grid.

        Active dates need not occur in the calendar. Dynamic counts therefore
        use resident membership rows, including extra assets and duplicate
        dates across assets, rather than a calendar rectangle.
        """
        coordinates = None
        try:
            if self._is_dynamic:
                assert self._active_membership is not None
                coordinates = self._active_membership.get_column(TIME)
                return _inclusive_date_count(coordinates, start, end)
            return _inclusive_date_count(self._times, start, end) * len(self._asset_ids)
        finally:
            coordinates = None

    def _contains_exact_keys(self, data: pl.DataFrame) -> bool:
        if data.height != self.size:
            return False
        keys = data.select(TIME, ASSET_ID).with_columns(
            normalize_date_expression(TIME, data.schema[TIME]),
            pl.col(ASSET_ID).cast(pl.String),
        )
        if self._is_dynamic:
            assert self._active_membership is not None
            return keys.sort([TIME, ASSET_ID]).equals(
                self._active_membership.select(TIME, ASSET_ID)
            )
        return bool(
            keys.select(
                (
                    pl.col(TIME).is_in(self._times.implode())
                    & pl.col(ASSET_ID).is_in(self._asset_ids.implode())
                ).all()
            ).item()
        )

    def grid_lazy(self) -> pl.LazyFrame:
        """Return active domain keys without materializing the grid."""

        if not self._is_dynamic:
            return self._all_grid_lazy()
        assert self._active_membership is not None
        return self._active_membership.lazy()

    def empty_frame(self) -> pl.DataFrame:
        return self.grid_lazy().with_columns(
            pl.lit(None, dtype=pl.Float64).alias(VALUE)
        ).collect()

    def align_lazy(
        self,
        data: pl.LazyFrame,
        *,
        trace_columns: Sequence[str] = (),
        _preserve_invalid_keys: bool = False,
    ) -> pl.LazyFrame:
        columns = [TIME, ASSET_ID, VALUE, *trace_columns]
        return (
            self.grid_lazy()
            .join(
                data.select(columns),
                on=[TIME, ASSET_ID],
                how="full" if _preserve_invalid_keys else "left",
                coalesce=True,
                maintain_order="left",
            )
        )

    def apply_membership_lazy(self, data: pl.LazyFrame, *, _preserve_invalid_keys: bool = False) -> pl.LazyFrame:
        invalid = pl.col(TIME).is_null() | pl.col(ASSET_ID).is_null()
        if not self._is_dynamic:
            selected = (
                pl.col(TIME).is_in(self._times.implode())
                & pl.col(ASSET_ID).is_in(self._asset_ids.implode())
            )
            return data.filter((selected | invalid) if _preserve_invalid_keys else selected)
        if _preserve_invalid_keys:
            # Keep malformed rows visible to deferred typed validation. A
            # single bounded left join avoids scanning the source a second
            # time merely to recover null keys discarded by an inner join.
            marker = "__core_active_key"
            columns = data.collect_schema().names()
            while marker in columns:
                marker += "_"
            assert self._active_membership is not None
            dates = self._active_membership.get_column(TIME).unique()
            selected = pl.col(TIME).is_in(dates.implode()) & pl.col(ASSET_ID).is_in(self._asset_ids.implode())
            # Preserve scan pushdown even though retaining invalid rows keeps
            # the join outer. Active keys may include off-calendar dates.
            bounded = data.filter(selected | invalid)
            return bounded.join(self.grid_lazy().with_columns(pl.lit(True).alias(marker)),
                on=[TIME, ASSET_ID], how="left", maintain_order="left").filter(
                    pl.col(marker).is_not_null() | invalid).drop(marker)
        return self.grid_lazy().join(
            data,
            on=[TIME, ASSET_ID],
            how="inner",
            maintain_order="left",
        )

    def normalize_frame(self, data: pl.DataFrame) -> pl.DataFrame:
        from bagelquant_core.frame import normalize_panel_frame

        normalized = normalize_panel_frame(data, numeric=False)
        return self.align_lazy(normalized.lazy()).collect()

    def apply_membership(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.apply_membership_lazy(data.lazy()).collect()

    def window(self, start, end) -> "Domain":
        """Select observation coordinates without changing an information cutoff."""
        from .preparation import active_preparation, domain_bytes
        from weakref import ref
        preparation = active_preparation()
        locator = (self, start, end)
        signature_locator = (ref(self), start, end)

        def build():
            dates = self._times.filter(self._times.is_between(start, end))
            if dates.is_empty():
                raise ValueError("calendar must contain at least one time")
            # Subsets of immutable normalized coordinates retain their key
            # types, uniqueness and order. Their identities still bind all
            # selected coordinates through the original canonical hashes.
            window = object.__new__(Domain)
            window._times = dates
            window.start_time, window.end_time = dates[0], dates[-1]
            window._is_dynamic = self._is_dynamic
            if self._is_dynamic:
                assert self._active_membership is not None
                window._active_membership = self._active_membership.filter(pl.col(TIME).is_between(start, end))
                window._asset_ids = window._active_membership[ASSET_ID].unique().sort()
            else:
                window._active_membership = None
                window._asset_ids = self._asset_ids
            window._set_signature(None if preparation is None else preparation.get("domain_signature", signature_locator))
            return window

        if preparation is None:
            return build()
        window = preparation.remember("domain_window", locator, build, domain_bytes)
        preparation.remember("domain_signature", signature_locator, lambda: window.signature, len)
        return window

    def _window_signature(self, start, end) -> str:
        """Retain exact coordinate evidence independently of its large window."""
        from .preparation import active_preparation
        from weakref import ref
        preparation = active_preparation()
        signature = None if preparation is None else preparation.get("domain_signature", (ref(self), start, end))
        return self.window(start, end).signature if signature is None else signature

    def equivalent_to(self, other: object) -> bool:
        return isinstance(other, Domain) and self.signature == other.signature

    def _grid(self) -> pl.DataFrame:
        return self.grid_lazy().collect()

    def _all_grid_lazy(self) -> pl.LazyFrame:
        return pl.DataFrame({TIME: self._times}).lazy().join(
            pl.DataFrame({ASSET_ID: self._asset_ids}).lazy(),
            how="cross",
        )

    def _normalize_dynamic_membership(
        self, universe: pl.DataFrame
    ) -> pl.DataFrame:
        missing = [
            column
            for column in (TIME, ASSET_ID, "active")
            if column not in universe.columns
        ]
        if missing:
            raise ValueError(
                f"dynamic universe is missing required columns: {missing}"
            )
        normalized = universe.select(TIME, ASSET_ID, "active").with_columns(
            normalize_date_expression(TIME, universe.schema[TIME]),
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("active").cast(pl.Boolean, strict=False),
        )
        if normalized.select(
            pl.any_horizontal(
                pl.col(TIME).is_null(), pl.col(ASSET_ID).is_null()
            ).any()
        ).item():
            raise ValueError("dynamic universe keys must be valid")
        if normalized.select(pl.col("active").is_null().any()).item():
            raise ValueError("dynamic universe active values must be valid")
        if normalized.select(
            pl.struct(TIME, ASSET_ID).is_duplicated().any()
        ).item():
            raise ValueError(
                "dynamic universe must be unique by (time, asset_id)"
            )
        return normalized.sort([TIME, ASSET_ID])
