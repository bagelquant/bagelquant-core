"""Bounded, metadata-admitted batches for new finite Core-owned results."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import date

import polars as pl

from .domain import _inclusive_date_count
from .node import ValueType
from .operator import OPERATOR_REGISTRY
from .operator.arithmetic import add, sub, mul, div
from .operator.basic import negate, pct_change
from .operator.normalization import rank
from .operator.rolling_stats import smooth
from .operator import rolling_stats as _rolling
from .operator.temporal import canonicalize_values, constant, project_domain
from .resources import ResourceLimits, active_resource_limits, resource_limits
from .store import CoreStore


_MIB = 1024 * 1024
_MAX_BLOCKS = 8
_START_MEMO_BYTES = 4096
_BUILTINS = {value.registry_name: (value, value.operation, value._plan_operation, value.contract)
             for value in (add, sub, mul, div, negate, pct_change, rank, smooth, constant,
                           canonicalize_values, project_domain)}
_SMOOTH_HELPERS = (_rolling._rolling_ewm_fw_expr, _rolling._smooth_trace,
                   _rolling.rolling_ewm_fw.operation, _rolling._plan_rolling_operation)


@dataclass(frozen=True)
class _ColdBatch:
    stop: int
    start: date
    end: date
    limits: ResourceLimits
    raw_bytes: int
    reserve_mib: int


class _ColdFiniteBatches:
    """Keep a cold invocation's own frontier distinct from retained history.

    Estimates are conservative selection rules, not hard native/RSS bounds.
    The controller retains metadata only, never input or result frames.
    """

    def __init__(self, node, direct, domain, policy, store, limits, stored_keys):
        self.eligible = False
        self._store, self._node_id = store, node.node_id
        self._parents = self._manifests = None
        self._keys = None
        self._schemas = {}
        self._read_starts: OrderedDict[tuple[int, date], date] = OrderedDict()
        self._parent_bounds: dict[str, tuple[int, int]] = {}
        self._output_asset_bytes = 0
        self.schema_reads = 0
        self._domain, self._limits = domain, limits
        self.planning_limits = limits
        self._owned: set[str] = set()
        self.planning_bytes = 0
        manifest = value = key = parent = header = manifests = None
        try:
            if (type(store) is not CoreStore or pl.__version__ != "1.42.1"
                    or not domain.is_dynamic or not domain.size or stored_keys is None
                    or set(stored_keys) != set(direct) or not direct
                    or store.inventory(node_id=node.node_id)):
                return
            expected = _BUILTINS.get(node.operator)
            if expected is None:
                return
            registered = OPERATOR_REGISTRY.get(node.operator)
            if (any(actual is not original for actual, original in zip(
                    (registered, registered.operation, registered._plan_operation, registered.contract), expected, strict=True))
                    or registered.contract.checkpoint_replay):
                return
            name = node.operator.rsplit(".", 1)[-1]
            if len(node.inputs) != (2 if name in {"add", "sub", "mul", "div"} else 1):
                return
            if name == "pct_change":
                if node.parameters != {"periods": 240} or policy["history_observations"] != 240:
                    return
            elif name == "smooth":
                if (node.parameters or policy["history_observations"] != 9
                        or (_rolling._SMOOTH_WINDOW, _rolling._SMOOTH_HALFLIFE,
                            _rolling._SMOOTH_MIN_PERIODS) != (10, 3.0, 5)
                        or (_rolling._rolling_ewm_fw_expr, _rolling._smooth_trace,
                            _rolling.rolling_ewm_fw.operation, _rolling._plan_rolling_operation) != _SMOOTH_HELPERS):
                    return
            elif name == "canonicalize_values":
                if (node.parameters not in ({}, {"significant_digits": 14})
                        or policy["history_observations"] != 0):
                    return
            elif name == "constant":
                if node.parameters != {"value": 100.0} or policy["history_observations"] != 0:
                    return
            elif node.parameters or policy["history_observations"] != 0:
                return
            if (node.panel_parameters and (name != "project_domain"
                    or set(node.panel_parameters) != {"membership"}
                    or len(node.panel_parameters["membership"]) != 1)):
                return
            manifests = {}
            self._output_asset_bytes = self._longest(domain)
            # This fixed allowance covers at most eight scalar Date entries,
            # their tuple/LRU objects and the small held parent-axis bounds.
            planning = (2 * domain.size * (64 + self._output_asset_bytes)
                + 256 * len(domain.times) + 65536 + _START_MEMO_BYTES)
            for parent, value in direct.items():
                key = stored_keys[parent]
                # A differently identified/in-memory binding is not stored authority.
                if (key is None or value.identity != key.identity or key.node_id != parent
                        or value.domain.signature != key.domain_identity
                        or value.value_type != ValueType.NUMERIC or not value.domain.is_dynamic
                        or not value.domain.times.equals(domain.times)
                        or name != "project_domain" and value.domain.signature != domain.signature):
                    return
                manifest = store.manifest(key.identity)
                if manifest is None:
                    raise ValueError("cold finite parent authority is missing")
                if (manifest["key"] != key.to_dict() or manifest["value_type"] != value.value_type.value
                        or tuple(manifest["trace_columns"]) != value.trace_columns
                        or manifest["trace_identity"] != value.trace_identity):
                    raise ValueError("cold finite parent differs from frozen authority")
                header = store.describe(key.identity)
                if header["calendar"] is None:
                    return  # Never construct a missing optional interval index.
                if (len(value.trace_columns) > 2 or len(manifest["partitions"]) > 4096
                        or manifest["artifacts"] or manifest["checkpoint"] is not None
                        or manifest["training_audits"]):
                    return
                manifests[parent] = manifest
                longest = self._longest(value.domain)
                self._parent_bounds[parent] = (len(value.domain.asset_ids), longest)
                planning += (2 * value.domain.size * (64 + longest)
                    + 256 * len(value.domain.times) + 4096 * len(manifest["partitions"]))
            planning_mib = (planning + _MIB - 1) // _MIB
            if (planning_mib > (limits.memory_target_mib - limits.cache_mib) // 4
                    or limits.memory_target_mib - planning_mib <= limits.cache_mib):
                return
            self.planning_bytes = planning
            self.planning_limits = replace(limits, memory_target_mib=limits.memory_target_mib - planning_mib)
            self._parents, self._manifests, self._keys = dict(direct), manifests, dict(stored_keys)
            self.eligible = True
        finally:
            manifest = value = key = parent = header = manifests = None
            del node, direct, domain, policy, store, limits, stored_keys

    @staticmethod
    def _longest(domain):
        return max((len(value.encode("utf-8")) for value in domain.asset_ids), default=0)

    def guard(self):
        assert self._store is not None or not self.eligible
        if self.eligible and {row["identity"] for row in self._store.inventory(node_id=self._node_id)} != self._owned:
            raise ValueError("cold finite authority changed during initialization; replan before continuing")

    def published(self, identity):
        if self.eligible:
            self._owned.add(identity)
            self.guard()

    def preflight_schema(self, check_canceled):
        """Admitted footer-only consumption, never a file-free prerequisite.

        The active R reservation includes P before this phase. Public lookup
        supplies plain immutable Core scan plans; custom Nodes cannot choose it.
        Unsupported schemas return to the original path before any value body;
        errors/cancellation propagate and are never caught for replay.
        """
        assert self.eligible and self._store is not None and self._keys is not None
        assert self._parents is not None
        planning_mib = (self.planning_bytes + _MIB - 1) // _MIB
        if active_resource_limits().memory_target_mib > self._limits.memory_target_mib - planning_mib:
            raise ValueError("cold finite schema preflight needs its admitted planning reservation")
        stored = schema = value = None
        try:
            for parent, value in self._parents.items():
                if parent in self._schemas:
                    continue
                check_canceled()
                stored = self._store.lookup(self._keys[parent], domain=value.domain, evidence=False)
                if stored is None:
                    raise ValueError("cold finite parent authority is missing")
                schema = stored.panel.lazy(include_traces=True).collect_schema()
                self.schema_reads += 1
                check_canceled()
                if (tuple(schema.names()) != ("time", "asset_id", "value", *value.trace_columns)
                        or schema["time"] != pl.Date or schema["asset_id"] != pl.String
                        or schema["value"] != pl.Float64
                        or any(schema[trace] != pl.Date for trace in value.trace_columns)):
                    self.eligible = False
                    self._schemas.clear()
                    return False
                self._schemas[parent] = schema
            return True
        finally:
            stored = schema = value = None
            del check_canceled

    def validate_frame(self, parent, frame):
        # The original public reader has already checked the actual selected
        # values, coordinates, nulls and traces; this adds no trusted shortcut.
        try:
            if frame.schema != self._schemas[parent]:
                raise ValueError("cold finite parent schema changed after admitted preflight")
        finally:
            del frame

    def _output_rows(self, start, end):
        """Count admitted keys, including implicit-null values, without a Domain."""
        assert self._domain is not None
        return self._domain._count_rows(start, end)

    def _numerical_start(self, origin, end, numerical_start):
        """Memoize successful scalar queries; never retain a callback or error."""
        value = None
        try:
            key = (origin, end)
            cached = self._read_starts.get(key)
            if cached is not None:
                self._read_starts.move_to_end(key)
                return cached
            value = numerical_start(origin, end)
            if type(value) is date:
                if len(self._read_starts) == _MAX_BLOCKS:
                    self._read_starts.popitem(last=False)
                self._read_starts[key] = value
            return value
        finally:
            value = None
            del numerical_start

    def plan(self, begin, calendar, numerical_start, canonical_sessions):
        """Admit public Domain metadata before considering any value allocation."""
        if not self.eligible:
            return None
        self.guard()
        assert self._parents is not None and self._manifests is not None and self._domain is not None
        limits = self._limits
        planning_mib = (self.planning_bytes + _MIB - 1) // _MIB
        manifest = value = None
        try:
            with resource_limits(replace(limits, memory_target_mib=limits.memory_target_mib - planning_mib)):
                checked = set()
                for count_blocks in range(_MAX_BLOCKS, 1, -1):
                    stop = min(begin + count_blocks * canonical_sessions, len(calendar))
                    if stop in checked:
                        continue
                    checked.add(stop)
                    # A final partial block must still be paired with another block.
                    origins = tuple(range(begin, stop, canonical_sessions))
                    if len(origins) < 2:
                        continue
                    end = calendar[stop - 1]
                    # Reject output row overflow before asset-local history
                    # queries or constructing/hashing any temporary Domain.
                    own_rows = self._output_rows(calendar[begin], end)
                    if own_rows > limits.batch_rows:
                        continue
                    start = min(self._numerical_start(origin,
                        calendar[min(origin + canonical_sessions, stop) - 1], numerical_start)
                        for origin in origins)
                    raw = self.planning_bytes + 8 * own_rows * (64 + self._output_asset_bytes + 8 * 2)
                    rows = 0
                    # Eligibility already proved every parent's calendar equal
                    # to this existing axis. Keep its exact inclusive count.
                    sessions = _inclusive_date_count(calendar, start, end)
                    for parent, value in self._parents.items():
                        # Global asset/string axes can overestimate a window,
                        # never understate its original dense rectangle charge.
                        assets, longest = self._parent_bounds[parent]
                        manifest = self._manifests[parent]
                        physical = sum(part["rows"] for part in manifest["partitions"]
                            if (part.get("end") is None or date.fromisoformat(part["end"]) >= start)
                            and (part.get("start") is None or date.fromisoformat(part["start"]) <= end))
                        # Dense preparation can exceed sparse admitted membership.
                        count = max(sessions * assets, physical)
                        rows += count
                        raw += 8 * count * (64 + longest + 8 * len(value.trace_columns))
                    reserve = (raw + _MIB - 1) // _MIB
                    if (rows <= _MAX_BLOCKS * limits.batch_rows
                            and reserve <= (limits.memory_target_mib - limits.cache_mib) // 4
                            and limits.memory_target_mib - reserve > limits.cache_mib):
                        return _ColdBatch(stop, start, end,
                            replace(limits, memory_target_mib=limits.memory_target_mib - reserve), raw, reserve)
            return None
        finally:
            manifest = value = None
            del calendar, numerical_start

    def close(self):
        self._parents = self._manifests = self._domain = self._store = None
        self._keys = None
        self._schemas.clear()
        self._read_starts.clear()
        self._parent_bounds.clear()
        self._output_asset_bytes = 0
        self._owned.clear()
        self.eligible = False

    def __enter__(self):
        return self

    def __exit__(self, _kind, _error, _traceback):
        self.close()
