"""Explicit trading-calendar and universe definitions for panel inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, normalize_asset_ids, normalize_time_series
from ..hashing import hash_dataframe, hash_mapping


class Domain:
    """Trading times and asset membership for long-form panel data."""

    def __init__(
        self,
        *,
        calendar: Sequence[Any] | pl.Series,
        universe: Sequence[Any] | pl.DataFrame,
    ) -> None:
        times = normalize_time_series(calendar)
        self.start_time = times[0]
        self.end_time = times[-1]
        self._times = times

        if isinstance(universe, pl.DataFrame):
            self._is_dynamic = True
            self._membership = self._normalize_dynamic_membership(universe)
        else:
            self._is_dynamic = False
            self._membership = self._static_membership(universe)

        self._asset_ids = (
            self._membership.select(ASSET_ID).unique().sort(ASSET_ID)[ASSET_ID]
        )
        self._grid_frame = self._make_grid(self._asset_ids)
        self._signature = hash_mapping(
            {
                "calendar": hash_dataframe(pl.DataFrame({TIME: self._times})),
                "start_time": str(self.start_time),
                "end_time": str(self.end_time),
                "membership": hash_dataframe(self._membership),
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
        return self._membership.clone()

    @property
    def is_dynamic(self) -> bool:
        return self._is_dynamic

    @property
    def signature(self) -> str:
        return self._signature

    def empty_frame(self) -> pl.DataFrame:
        return self._grid().with_columns(pl.lit(None, dtype=pl.Float64).alias(VALUE))

    def normalize_frame(self, data: pl.DataFrame) -> pl.DataFrame:
        from ..frame import normalize_panel_frame

        normalized = normalize_panel_frame(data, numeric=False)
        grid = self._grid()
        aligned = grid.join(normalized, on=[TIME, ASSET_ID], how="left")
        return self.apply_membership(aligned)

    def apply_membership(self, data: pl.DataFrame) -> pl.DataFrame:
        if not self._is_dynamic:
            return data
        return (
            data.join(self._membership, on=[TIME, ASSET_ID], how="left")
            .filter(pl.col("active").fill_null(False))
            .drop("active")
        )

    def equivalent_to(self, other: object) -> bool:
        return isinstance(other, Domain) and self.signature == other.signature

    def _grid(self) -> pl.DataFrame:
        return self._grid_frame

    def _make_grid(self, assets: pl.Series) -> pl.DataFrame:
        return pl.DataFrame({TIME: self._times}).join(
            pl.DataFrame({ASSET_ID: assets}),
            how="cross",
        )

    def _static_membership(self, universe: Sequence[Any]) -> pl.DataFrame:
        assets = normalize_asset_ids(universe)
        return (
            self._make_grid(assets)
            .with_columns(pl.lit(True).alias("active"))
        )

    def _normalize_dynamic_membership(self, universe: pl.DataFrame) -> pl.DataFrame:
        missing = [
            column
            for column in (TIME, ASSET_ID, "active")
            if column not in universe.columns
        ]
        if missing:
            raise ValueError(f"dynamic universe is missing required columns: {missing}")
        normalized = universe.select(TIME, ASSET_ID, "active").with_columns(
            pl.col(TIME).cast(pl.Date, strict=False),
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("active").cast(pl.Boolean, strict=False),
        )
        if normalized.select(
            pl.any_horizontal(pl.col(TIME).is_null(), pl.col(ASSET_ID).is_null()).any()
        ).item():
            raise ValueError("dynamic universe keys must be valid")
        if normalized.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
            raise ValueError("dynamic universe must be unique by (time, asset_id)")
        assets = normalize_asset_ids(
            normalized.select(ASSET_ID).unique().sort(ASSET_ID)[ASSET_ID]
        )
        grid = self._make_grid(assets)
        return (
            grid.join(normalized, on=[TIME, ASSET_ID], how="left")
            .with_columns(pl.col("active").fill_null(False))
            .sort([TIME, ASSET_ID])
        )
