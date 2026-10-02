"""Register composer calling conventions in the unified operator catalog."""
from __future__ import annotations
from collections.abc import Callable
import polars as pl
from ..operator import Operator, OperationCatalog
from ..operation_contract import OperationContract

COMPOSER_REGISTRY = OperationCatalog("composer")

def composer(operation: Callable[..., pl.DataFrame] | None = None, *,
               contract: OperationContract | None = None):
    """Decorate a numerical function as a registered Operator."""
    def decorate(func: Callable[..., pl.DataFrame]) -> Operator:
        wrapped = Operator(func, contract=contract, input_mode="composer",
                           minimum_inputs=2, maximum_inputs=None)
        COMPOSER_REGISTRY.add(wrapped.registry_name, wrapped)
        return wrapped
    return decorate(operation) if operation is not None else decorate


def _horizontal_value_plan(
    frames: tuple[pl.LazyFrame, ...],
) -> tuple[pl.LazyFrame, list[pl.Expr]]:
    """Combine proven positionally aligned values without hashing keys."""

    columns = [f"__value_{index}" for index in range(len(frames))]
    plans = [
        frames[0].rename({"value": columns[0]}),
        *[
            frame.select(pl.col("value").alias(column))
            for frame, column in zip(
                frames[1:],
                columns[1:],
                strict=True,
            )
        ],
    ]
    try:
        combined = pl.concat(plans, how="horizontal_extend")
    except ValueError as error:
        if "horizontal_extend" not in str(error):
            raise
        combined = pl.concat(plans, how="horizontal")
    return combined, [pl.col(column) for column in columns]


def _horizontal_expression_plan(
    frames: tuple[pl.LazyFrame, ...],
    expression: pl.Expr,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    combined, _ = _horizontal_value_plan(frames)
    return (
        combined.select(
            "time",
            "asset_id",
            expression.alias("value"),
        ),
        order,
        asset_time_ordered,
    )
