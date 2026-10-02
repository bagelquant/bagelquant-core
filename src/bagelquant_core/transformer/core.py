"""Register transformer calling conventions in the unified operator catalog."""
from __future__ import annotations
from collections.abc import Callable
import polars as pl
from ..operator import Operator, OperationCatalog
from ..operation_contract import OperationContract

TRANSFORMER_REGISTRY = OperationCatalog("transformer")

def transformer(operation: Callable[..., pl.DataFrame] | None = None, *,
               contract: OperationContract | None = None):
    """Decorate a numerical function as a registered Operator."""
    def decorate(func: Callable[..., pl.DataFrame]) -> Operator:
        wrapped = Operator(func, contract=contract, input_mode="transformer",
                           minimum_inputs=1, maximum_inputs=1)
        TRANSFORMER_REGISTRY.add(wrapped.registry_name, wrapped)
        return wrapped
    return decorate(operation) if operation is not None else decorate


def _ordered_expression_plan(
    frame: pl.LazyFrame,
    expression: pl.Expr,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    """Apply an asset-time expression without redundant physical sorting."""

    source = frame if asset_time_ordered else frame.sort(["asset_id", "time"])
    output_order = order if asset_time_ordered else "asset_time"
    return (
        source.with_columns(expression.alias("value")).select(
            "time",
            "asset_id",
            "value",
        ),
        output_order,
        True,
    )


def _expression_plan(
    frame: pl.LazyFrame,
    expression: pl.Expr,
    order: str | None,
    asset_time_ordered: bool,
) -> tuple[pl.LazyFrame, str | None, bool]:
    """Apply an order-independent expression without sorting keys."""

    return (
        frame.with_columns(expression.alias("value")).select(
            "time",
            "asset_id",
            "value",
        ),
        order,
        asset_time_ordered,
    )
