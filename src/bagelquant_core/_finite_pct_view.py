"""Optional, operation-local membership for the unchanged percentage kernel."""
from __future__ import annotations

from dataclasses import replace
from types import TracebackType
from typing import Self

import polars as pl

from .domain import Domain
from .hashing import hash_mapping
from .node import Node, ValueType
from .operator import OPERATOR_REGISTRY
from .operator.basic import pct_change
from .resources import ResourceLimits, active_resource_limits


_MIB = 1024 * 1024
_PCT = "bagelquant_core.operator.basic.pct_change"
_OPERATION = pct_change.operation
_PLAN = pct_change._plan_operation
_CONTRACT = pct_change.contract


class _PctKernelView:
    """Charge B-only work before allocation; keep original proof/read scopes.

    The charge is conservative admission, not a native RSS bound. Only our own
    aliases are released; foreign exception frames may retain their arguments.
    """

    def __init__(self, node, scoped: dict[str, Node], output_domain: Domain,
                 history: int, limits: ResourceLimits):
        self.limits = limits
        self.admitted = False
        self.closed = False
        self.raw_bytes = self.reserve_mib = 0
        self._inputs: dict[str, Node] | None = None
        try:
            if (node.operator != _PCT or len(node.inputs) != 1 or node.panel_parameters
                    or len(scoped) != 1 or isinstance(history, bool)
                    or not isinstance(history, int) or history <= 0
                    or not output_domain.is_dynamic or output_domain.size == 0):
                return
            registered = OPERATOR_REGISTRY.get(node.operator)
            if (registered is not pct_change or registered.operation is not _OPERATION
                    or registered._plan_operation is not _PLAN or registered.contract is not _CONTRACT):
                return
            source = next(iter(scoped.values()))
            if not source.domain.is_dynamic or source.value_type != ValueType.NUMERIC:
                return
            schema = source.lazy(include_traces=True).collect_schema()
            if (schema["value"] != pl.Float64
                    or any(schema[name] != pl.Date for name in source.trace_columns)):
                return
            # Scalars/held axes only: never expand membership or collect values here.
            assets = source.domain.asset_ids
            longest = max((len(asset.encode("utf-8")) for asset in assets), default=0)
            self.raw_bytes = (6 * source.domain.size * (64 + longest + 8 * len(source.trace_columns))
                + 8 * len(assets) + 256 * len(source.domain.times) + 65536)
            reservation = (self.raw_bytes + _MIB - 1) // _MIB
            if (reservation > (limits.memory_target_mib - limits.cache_mib) // 4
                    or limits.memory_target_mib - reservation <= limits.cache_mib):
                return
            self.reserve_mib = reservation
            self.limits = replace(limits, memory_target_mib=limits.memory_target_mib - reservation)
            self.admitted = True
        finally:
            source = schema = assets = None
            del node, scoped, output_domain

    def __enter__(self) -> Self:
        if self.closed:
            raise ValueError("percentage kernel view is closed")
        return self

    def prepare(self, scoped: dict[str, Node], output_domain: Domain,
                output_start, end, history: int, check_canceled) -> dict[str, Node]:
        """Validate the full ordinary dense input before selecting any keys."""
        source = dense = output = assets = preceding = keys = domain = frame = value = None
        try:
            if not self.admitted:
                return scoped
            if self.closed or active_resource_limits() != self.limits:
                raise ValueError("percentage kernel view needs its admitted resource scope")
            check_canceled()
            parent, source = next(iter(scoped.items()))
            # Public collection retains implicit null membership positions and
            # validates the entire selected input, including discarded rows.
            dense = source.collect(dense=True, include_traces=True)
            check_canceled()
            output = output_domain.grid_lazy().filter(
                pl.col("time").is_between(output_start, end)).collect()
            # A reused partial prefix can leave an empty tail in a nonempty
            # canonical block. Keep its original runtime without dummy keys.
            if output.is_empty():
                return scoped
            assets = output.select("asset_id").unique()
            preceding = (dense.select("time", "asset_id")
                .filter(pl.col("time") < output_start)
                .join(assets, on="asset_id", how="semi")
                .group_by("asset_id", maintain_order=True).tail(history)
                .select("time", "asset_id"))
            keys = pl.concat([preceding, output]).sort("time", "asset_id")
            domain = Domain(calendar=source.domain.times,
                universe=keys.with_columns(pl.lit(True).alias("active")))
            frame = keys.join(dense, on=["time", "asset_id"], how="left", maintain_order="left")
            token = hash_mapping({"input_proof": source.identity, "kernel_domain": domain.signature})
            value = Node.from_domain(frame, domain, identity=token, value_type=source.value_type,
                trace_columns=source.trace_columns,
                trace_identity=(hash_mapping({"trace_proof": source.trace_identity,
                    "kernel_domain": domain.signature, "columns": source.trace_columns})
                    if source.trace_columns else None))
            self._inputs = {parent: value}
            check_canceled()
            return self._inputs
        finally:
            # Do not clear public constructors/library/caller traceback frames.
            source = dense = output = assets = preceding = keys = domain = frame = value = None
            del scoped, output_domain, output_start, end, history, check_canceled

    def close(self) -> None:
        self._inputs = None
        self.closed = True

    def __exit__(self, _kind: type[BaseException] | None, _error: BaseException | None,
                 _traceback: TracebackType | None) -> None:
        self.close()
