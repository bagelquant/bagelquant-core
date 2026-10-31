"""Finite, deterministic cross-sectional descriptors and residuals."""
from __future__ import annotations

import numpy as np
import polars as pl

from bagelquant_core.frame import TIME, ASSET_ID, VALUE
from bagelquant_core.operation_contract import ExecutionMode, OperationContract, TraceRule
from ._definition import operator


def mad_standardize_values(values, *, clip_multiplier=5.0, mad_scale=1.4826, min_scale=1e-12):
    """Clip finite observations at median/MAD and use population standard deviation."""
    if not np.isfinite([clip_multiplier, mad_scale, min_scale]).all() or min(clip_multiplier, mad_scale, min_scale) <= 0:
        raise ValueError("MAD scale, multiplier and minimum scale must be finite and positive")
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if not valid.any():
        return result
    selected = values[valid].copy()
    median = np.median(selected)
    mad = np.median(np.abs(selected - median))
    if mad > 0:
        selected = np.clip(selected, median - clip_multiplier * mad_scale * mad,
                           median + clip_multiplier * mad_scale * mad)
    scale = selected.std(ddof=0)
    if scale > min_scale:
        result[valid] = (selected - selected.mean()) / scale
    return result


def ols_residual_values(values, controls, *, fit_intercept=True):
    """Unweighted full-rank OLS residuals with strictly more samples than parameters."""
    values, controls = np.asarray(values, dtype=float), np.asarray(controls, dtype=float)
    if values.ndim != 1 or controls.ndim != 2 or len(values) != len(controls) or not controls.shape[1]:
        raise ValueError("residuals require one target and a nonempty aligned controls matrix")
    valid = np.isfinite(values) & np.isfinite(controls).all(axis=1)
    result = np.full(values.shape, np.nan)
    design = controls[valid]
    if fit_intercept:
        design = np.column_stack([np.ones(valid.sum()), design])
    if valid.sum() > design.shape[1] and np.linalg.matrix_rank(design) == design.shape[1]:
        result[valid] = values[valid] - design @ np.linalg.lstsq(design, values[valid], rcond=None)[0]
    return result


_CROSS_SECTION = OperationContract(execution=ExecutionMode.EAGER_BARRIER,
    trace_rule=TraceRule.CROSS_SECTION_MAX, history=0)


@operator(contract=_CROSS_SECTION, version="2")
def mad_zscore(frame: pl.DataFrame, *, clip_multiplier: float = 5.0,
               mad_scale: float = 1.4826, min_scale: float = 1e-12) -> pl.DataFrame:
    """Finite median/MAD clipping followed by population cross-sectional z-score."""
    pieces = [part.select(TIME, ASSET_ID).with_columns(pl.Series(VALUE,
        mad_standardize_values(part[VALUE].to_numpy(), clip_multiplier=clip_multiplier,
            mad_scale=mad_scale, min_scale=min_scale)).fill_nan(None))
        for part in frame.sort(TIME, ASSET_ID).partition_by(TIME, maintain_order=True)]
    return pl.concat(pieces) if pieces else frame.select(TIME, ASSET_ID, VALUE)


@operator(contract=_CROSS_SECTION, version="2")
def cross_sectional_residual(frame: pl.DataFrame, *, controls: tuple[pl.DataFrame, ...],
                             fit_intercept: bool = True) -> pl.DataFrame:
    """Cross-sectional OLS residuals; incomplete or rank-deficient samples are null."""
    if not controls:
        raise ValueError("residuals require at least one control")
    names = [f"control_{index}" for index in range(len(controls))]
    data = frame.select(TIME, ASSET_ID, VALUE)
    for control, name in zip(controls, names, strict=True):
        data = data.join(control.select(TIME, ASSET_ID, pl.col(VALUE).alias(name)),
            on=[TIME, ASSET_ID], how="left")
    pieces = [part.select(TIME, ASSET_ID).with_columns(pl.Series(VALUE,
        ols_residual_values(part[VALUE].to_numpy(), part.select(names).to_numpy(),
            fit_intercept=fit_intercept)).fill_nan(None))
        for part in data.sort(TIME, ASSET_ID).partition_by(TIME, maintain_order=True)]
    return pl.concat(pieces) if pieces else frame.select(TIME, ASSET_ID, VALUE)


@operator(contract=_CROSS_SECTION, version="2", auxiliary_types={"weights": "numeric"})
def cross_sectional_weighted_mean(frame: pl.DataFrame, *, weights: pl.DataFrame,
                                  positive_weights: bool = True) -> pl.DataFrame:
    """Broadcast the finite weighted daily mean across the input coordinates."""
    joined = frame.select(TIME, ASSET_ID, VALUE).join(
        weights.select(TIME, ASSET_ID, pl.col(VALUE).alias("weight")),
        on=[TIME, ASSET_ID], how="left")
    valid = pl.col(VALUE).is_finite().fill_null(False) & pl.col("weight").is_finite().fill_null(False)
    if positive_weights:
        valid &= pl.col("weight") > 0
    means = joined.filter(valid).group_by(TIME).agg(
        ((pl.col(VALUE) * pl.col("weight")).sum() / pl.col("weight").sum()).alias("mean"))
    return frame.select(TIME, ASSET_ID).join(means, on=TIME, how="left").select(
        TIME, ASSET_ID, pl.when(pl.col("mean").is_finite()).then(pl.col("mean")).otherwise(None).alias(VALUE))


__all__ = ["mad_zscore", "cross_sectional_residual", "cross_sectional_weighted_mean",
           "mad_standardize_values", "ols_residual_values"]
