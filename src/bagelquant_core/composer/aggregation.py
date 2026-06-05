"""Multi-input aggregation composers."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import numpy as np
import pandas as pd

from .core import composer


def _validate_frames(frames: tuple[pd.DataFrame, ...], *, operation: str) -> None:
    if not frames:
        raise ValueError(f"{operation} requires at least one frame")


def _validate_weights(
    frames: tuple[pd.DataFrame, ...],
    *,
    weights: Sequence[Real],
    operation: str,
) -> tuple[Real, ...]:
    _validate_frames(frames, operation=operation)
    if isinstance(weights, (str, bytes)):
        raise TypeError(f"{operation} weights must be a sequence of real numbers")
    if len(weights) != len(frames):
        raise ValueError(f"{operation} requires one weight per frame")
    if any(
        not isinstance(weight, Real) or isinstance(weight, bool)
        for weight in weights
    ):
        raise TypeError(f"{operation} weights must be real numbers")
    return tuple(weights)


@composer
def sum_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Add one or more frames."""

    _validate_frames(frames, operation="sum_frames")
    output = frames[0].copy()
    for frame in frames[1:]:
        output = output.add(frame)
    return output


@composer
def mean(*frames: pd.DataFrame) -> pd.DataFrame:
    """Return the arithmetic mean of one or more frames."""

    _validate_frames(frames, operation="mean")
    return sum_frames.operation(*frames).div(len(frames))


@composer
def product(*frames: pd.DataFrame) -> pd.DataFrame:
    """Multiply one or more frames."""

    _validate_frames(frames, operation="product")
    output = frames[0].copy()
    for frame in frames[1:]:
        output = output.mul(frame)
    return output


@composer
def minimum(*frames: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise minimum values across one or more frames."""

    _validate_frames(frames, operation="minimum")
    output = frames[0].copy()
    for frame in frames[1:]:
        output = pd.DataFrame(
            np.minimum(output.to_numpy(), frame.to_numpy()),
            index=output.index,
            columns=output.columns,
        )
    return output


@composer
def maximum(*frames: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise maximum values across one or more frames."""

    _validate_frames(frames, operation="maximum")
    output = frames[0].copy()
    for frame in frames[1:]:
        output = pd.DataFrame(
            np.maximum(output.to_numpy(), frame.to_numpy()),
            index=output.index,
            columns=output.columns,
        )
    return output


@composer
def weighted_sum(
    *frames: pd.DataFrame,
    weights: Sequence[Real],
) -> pd.DataFrame:
    """Return the weighted sum of one or more frames."""

    checked_weights = _validate_weights(
        frames,
        weights=weights,
        operation="weighted_sum",
    )
    output = frames[0] * checked_weights[0]
    for frame, weight in zip(frames[1:], checked_weights[1:]):
        output = output.add(frame * weight)
    return output


@composer
def weighted_mean(
    *frames: pd.DataFrame,
    weights: Sequence[Real],
) -> pd.DataFrame:
    """Return the weighted mean of one or more frames."""

    checked_weights = _validate_weights(
        frames,
        weights=weights,
        operation="weighted_mean",
    )
    total_weight = sum(checked_weights)
    if total_weight == 0:
        raise ValueError("weighted_mean weights must not sum to zero")
    return weighted_sum.operation(*frames, weights=checked_weights).div(total_weight)


min = minimum
max = maximum
