"""Explicit trading-calendar and universe definitions for panel inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, normalize_asset_ids, normalize_time_series
from ..hashing import hash_dataframe, hash_mapping


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

        self._signature = identity or hash_mapping(
            {
                "calendar": hash_dataframe(pl.DataFrame({TIME: self._times})),
                "assets": hash_dataframe(
                    pl.DataFrame({ASSET_ID: self._asset_ids})
                ),
                "dynamic": self._is_dynamic,
                "active": (
                    None
                    if self._active_membership is None
                    else hash_dataframe(self._active_membership)
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

    def _contains_exact_keys(self, data: pl.DataFrame) -> bool:
        if data.height != self.size:
            return False
        keys = data.select(TIME, ASSET_ID).with_columns(
            pl.col(TIME).cast(pl.Date, strict=False),
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
    ) -> pl.LazyFrame:
        columns = [TIME, ASSET_ID, VALUE, *trace_columns]
        return (
            self.grid_lazy()
            .join(
                data.select(columns),
                on=[TIME, ASSET_ID],
                how="left",
                maintain_order="left",
            )
        )

    def apply_membership_lazy(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if not self._is_dynamic:
            return data.filter(
                pl.col(TIME).is_in(self._times.implode())
                & pl.col(ASSET_ID).is_in(self._asset_ids.implode())
            )
        return self.grid_lazy().join(
            data,
            on=[TIME, ASSET_ID],
            how="inner",
            maintain_order="left",
        )

    def normalize_frame(self, data: pl.DataFrame) -> pl.DataFrame:
        from ..frame import normalize_panel_frame

        normalized = normalize_panel_frame(data, numeric=False)
        return self.align_lazy(normalized.lazy()).collect()

    def apply_membership(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.apply_membership_lazy(data.lazy()).collect()

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
            pl.col(TIME).cast(pl.Date, strict=False),
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("active").cast(pl.Boolean, strict=False),
        )
        if normalized.select(
            pl.any_horizontal(
                pl.col(TIME).is_null(), pl.col(ASSET_ID).is_null()
            ).any()
        ).item():
            raise ValueError("dynamic universe keys must be valid")
        if normalized.select(
            pl.struct(TIME, ASSET_ID).is_duplicated().any()
        ).item():
            raise ValueError(
                "dynamic universe must be unique by (time, asset_id)"
            )
        return normalized.sort([TIME, ASSET_ID])
