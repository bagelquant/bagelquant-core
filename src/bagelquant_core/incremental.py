"""Canonical finite-history blocks, independent of application scheduling.

Every first build and rebuild uses the same 32-session mathematical blocks.
Block reuse is proven from all parent values, coordinates, membership and traces.
Unbounded/custom contracts conservatively replay their required full history.
"""
from __future__ import annotations

from datetime import date, timedelta
from dataclasses import replace
from typing import Mapping

import numpy as np
import polars as pl

from .domain import Domain
from .execution import ExecutionRuntime
from .graph import Graph
from .graph_management import parents
from .hashing import hash_mapping
from .logical import LogicalGraphSpec
from .materialization import MaterializationKey, NodeMaterialization, materialization_trace_identity
from .node import Node
from .operator import OPERATOR_REGISTRY

CANONICAL_SESSIONS = 32
FINITE_EXECUTION_POLICY = "finite_causal_blocks.v3"


def _validate_declared_calendar_axes(spec, inputs, completed, parameters):
    """Check explicit calendar axes before upstream finite mathematical work."""
    if not any(node.node_type == "operator" and
            OPERATOR_REGISTRY.get(node.operator).contract.checkpoint_calendar_role is not None
            for node in spec.nodes):
        return
    bound = Graph.from_logical_spec(spec, inputs=inputs, node_bindings=completed,
                                   parameter_bindings=parameters)
    for node in bound.nodes:
        if node.node_type != "operator" or node.contract.checkpoint_calendar_role is None:
            continue
        calendars = node.spec_panel_parameters().get(node.contract.checkpoint_calendar_role, ())
        if len(calendars) != 1:
            raise ValueError("checkpoint calendar role requires one explicit auxiliary Node")
        days = calendars[0].domain.times
        if not days.filter(days.is_between(node.domain.times.min(), node.domain.times.max())).equals(node.domain.times):
            raise ValueError("explicit checkpoint calendar axis differs from actual execution calendar")


def finite_execution_policies(spec, *, anchor):
    policies = {}
    for node in spec.nodes:
        if node.node_type != "operator":
            continue
        registered = OPERATOR_REGISTRY.get(node.operator)
        history = registered.history(node.parameters)
        if history is not None:
            policies[node.node_id] = {"name": FINITE_EXECUTION_POLICY,
                "sessions": CANONICAL_SESSIONS, "anchor": str(anchor),
                "history_observations": history}
    return policies


def _canonical_halos(domain, observations):
    calendar = domain.times.cast(pl.Int32).to_numpy()
    if not len(calendar):
        return {}
    origins = np.arange(0, len(calendar), CANONICAL_SESSIONS)
    earliest = calendar[np.maximum(0, origins-observations)].copy()
    if observations and domain.is_dynamic:
        membership = domain.grid_lazy().sort("asset_id", "time").collect()
        times = membership["time"].cast(pl.Int32).to_numpy()
        lengths = membership["asset_id"].rle().struct.field("len").to_numpy()
        offset = 0
        for length in lengths:
            asset_times = times[offset:offset+int(length)]
            offset += int(length)
            counts = np.searchsorted(asset_times, calendar[origins], side="left")
            present = counts > 0
            earliest[present] = np.minimum(earliest[present], asset_times[np.maximum(0, counts[present]-observations)])
    earliest[0] = calendar[0]
    epoch = date(1970, 1, 1)
    return {int(origin): epoch+timedelta(days=int(day)) for origin, day in zip(origins, earliest, strict=True)}


def _numerical_read_start(domain: Domain, output_start: date, end: date, history: int) -> date:
    """Keep prior admitted keys only for assets occurring in missing output.

    The original canonical proof horizon remains separate. Use this only for
    kernels whose values and traces are exactly independent of earlier state.
    Null values remain admitted observations; reentry retains historical keys.
    """
    if history == 0:
        return output_start
    calendar = domain.times.to_list()
    earliest = calendar[max(0, calendar.index(output_start) - history)]
    if domain.is_dynamic:
        membership = domain.grid_lazy()
        needed = membership.filter(pl.col('time').is_between(output_start, end)).select('asset_id').unique()
        first = (membership.filter(pl.col('time') < output_start)
                 .join(needed, on='asset_id', how='semi')
                 .group_by('asset_id').agg(pl.col('time').top_k(history).min().alias('first'))
                 .select(pl.col('first').min()).collect().item())
        if first is not None:
            earliest = min(earliest, first)
    return earliest


_EXACT_WINDOW_OPERATORS = frozenset((
    'bagelquant_core.operator.basic.pct_change',
    'bagelquant_core.operator.rolling_stats.smooth',
))


def scoped_domain(domain, start, end, *, identity=None):
    dates = domain.times.filter(domain.times.is_between(start, end))
    universe = (domain.grid_lazy().filter(pl.col("time").is_between(start, end)).collect().with_columns(
        pl.lit(True).alias("active")) if domain.is_dynamic else domain.asset_ids)
    return Domain(calendar=dates, universe=universe, identity=identity)


def _relevant_parameters(node, bindings):
    used = set()
    def visit(value):
        if isinstance(value, str) and value.startswith("$"):
            used.add(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    visit(node.parameters)
    return {name: bindings[name] for name in sorted(used) if name in bindings}


def _reference_node(frame, domain, key, *, value_type, trace_columns):
    """Compose already typed original channels without reading old file schemas."""
    return Node._from_plan(frame, domain=domain, name=key.node_id, metadata=None,
        identity=key.identity, dense_output=False, value_type=value_type,
        trace_columns=trace_columns, trace_identity=materialization_trace_identity(key, trace_columns))


def _same_finite_policy(saved, policy):
    return {name: value for name, value in saved.items() if name != "parameter_bindings"} == policy


def _finite_partials(store, node_id, implementation_id, policy):
    partials = {}
    for header in store.materialization_candidates(node_id, implementation_id, policy_key=hash_mapping(policy)):
        partials.setdefault(header["coverage_start"], []).append({**header,
            "key": {"context_identity": header["context_identity"]}})
    return partials


def _checkpoint_parent_scopes(node, contract, config, domain, direct, end, *, scoped=False):
    """One owner route for actual consumed parent endpoints and fit cutoffs."""
    declared = ({} if contract.checkpoint_parent_scopes is None else
        contract.checkpoint_parent_scopes(config, tuple(domain.times.to_list()), end))
    ends = {parent: str(end) for parent in direct}
    cutoffs = {}
    for role, (role_end, cutoff) in declared.items():
        for parent in node.panel_parameters.get(role, ()):
            if parent not in node.inputs and not direct[parent].trace_columns:
                ends[parent] = None if role_end is None else str(role_end)
                if scoped and cutoff is not None:
                    cutoffs[parent] = str(cutoff)
    return ends, cutoffs


def _checkpoint_candidates(store, node, key, domain, parameters, parent_scopes, *, calendar_role=None, anchor_calendar=()):
    relevant = _relevant_parameters(node, parameters)
    for item in store.materialization_candidates(node.node_id, key.implementation_id,
            policy_name="checkpoint_prefix.v2", parameter_key=hash_mapping(relevant),
            coverage_start=str(domain.times.min()), through=str(domain.times.max()), checkpoint=True):
        manifest = store.manifest(item["identity"])
        policy = manifest.get("execution_policy") or {}
        if (policy.get("name") != "checkpoint_prefix.v2" or "parameters" not in policy
                or _relevant_parameters(node, policy["parameters"]) != relevant
                or manifest["coverage_start"] != str(domain.times.min())
                or manifest["through"] > str(domain.times.max()) or not manifest.get("checkpoint")):
            continue
        end = date.fromisoformat(manifest["through"])
        if calendar_role is not None and (policy.get("calendar_role") != calendar_role
                or policy.get("anchor_calendar") != [str(day) for day in anchor_calendar if day <= end]):
            continue
        ends, cutoffs = parent_scopes(end)
        if "parent_ends" in policy and policy["parent_ends"] != ends:
            continue
        if policy.get("parent_cutoffs") is not None and policy["parent_cutoffs"] != cutoffs:
            continue
        yield manifest


def execute_finite_node(spec, node_id, *, inputs: Mapping[str, Node], completed: Mapping[str, Node],
                        parameters, key, domain, policy, store, check_canceled, input_proof=None,
                        input_window_loader=None, completed_materializations=()):
    """Execute one admitted node, verifying content proofs before reusing blocks."""
    declared = {node.node_id: node for node in spec.nodes}
    node = declared[node_id]
    _validate_declared_calendar_axes(spec, inputs, completed, parameters)
    relevant_parameters = _relevant_parameters(node, parameters)
    publication_policy = {**policy, "parameter_bindings": relevant_parameters}
    def same_policy(saved):
        return _same_finite_policy(saved, policy)
    def contexts(start, end):
        return tuple(dict.fromkeys(hash_mapping({"block": policy, "start": start,
            "end": end, "parameters": bindings}) for bindings in (relevant_parameters, parameters)))
    direct = {parent: (inputs[declared[parent].input_key] if declared[parent].node_type == "input"
                       else completed[parent]) for parent in parents(node)}
    calendar = domain.times.to_list()
    if not calendar:
        raise ValueError("finite update requires an explicit nonempty calendar")
    if date.fromisoformat(policy["anchor"]) > calendar[0]:
        raise ValueError("block anchor must precede the first observation session")
    # Membership halos count active coordinates, including long universe gaps.
    halos = _canonical_halos(domain, policy["history_observations"])
    local = LogicalGraphSpec({node_id: node_id}, spec.nodes)
    blocks = []
    references = []
    computed = reused = 0
    partials = _finite_partials(store, node_id, key.implementation_id, policy)
    completed_prefixes = [saved for identity in completed_materializations
        if (saved := store.describe(identity)) is not None
        and saved["key"]["implementation_id"] == key.implementation_id
        and saved["coverage_start"] == str(calendar[0])]
    matching_prefixes = [saved for saved in completed_prefixes
        if same_policy(saved["execution_policy"] or {})
        and ("parameter_bindings" not in saved["execution_policy"]
             or saved["execution_policy"]["parameter_bindings"] == relevant_parameters)]
    # An obsolete receipt is retained historical evidence, not a veto over an
    # independently completed current-policy authority. The latter still needs
    # matching block/parameter/input proofs below before any kernel may run.
    if not matching_prefixes and any(not same_policy(saved["execution_policy"] or {})
            for saved in completed_prefixes):
        raise ValueError("Completed finite history has no matching execution-policy proof; "
            "run explicit policy-proof maintenance before updating")
    require_interval_proof = bool(matching_prefixes)
    legacy_completed = any("parameter_bindings" not in saved["execution_policy"]
        for saved in matching_prefixes)
    if legacy_completed and not any(item["key"]["context_identity"] in contexts(item["coverage_start"], item["through"])
            for items in partials.values() for item in items):
        raise ValueError("Completed finite history has no relevant parameter proof; "
            "run explicit parameter-proof maintenance before updating")
    if require_interval_proof:
        # Check the completed prefix before any kernel runs. A new opaque proof
        # namespace (or changes covering every saved block) cannot masquerade
        # as a permitted incremental update and replay the whole history.
        reusable = False
        for begin in range(0, len(calendar), CANONICAL_SESSIONS):
            warm = halos[begin]
            for candidate in partials.get(str(calendar[begin]), ()):
                prior_end = date.fromisoformat(candidate["through"])
                if prior_end not in calendar[begin:begin + CANONICAL_SESSIONS]:
                    continue
                proofs = []
                for parent, value in direct.items():
                    token = None if input_proof is None else input_proof(parent, value, warm, prior_end)
                    if token is None:
                        raise ValueError("Incremental update needs interval evidence for completed inputs; "
                            "run explicit owner index maintenance or supply input_proof before updating")
                    proofs.append(hash_mapping({"version": FINITE_EXECUTION_POLICY, "content": token,
                        "domain": scoped_domain(value.domain, warm, prior_end).signature,
                        "type": value.value_type.value, "traces": value.trace_columns}))
                own = scoped_domain(domain, calendar[begin], prior_end)
                if any(MaterializationKey(node_id, key.implementation_id, tuple(proofs), own.signature, context).identity
                        == candidate["identity"] for context in contexts(str(calendar[begin]), str(prior_end))):
                    reusable = True
                    break
            if reusable:
                break
        if not reusable:
            raise ValueError("Input proof changes cover the completed finite history; "
                "plan affected intervals or run explicit proof maintenance before updating")
    computed_sessions = 0
    kernel_sessions = 0
    input_sessions = 0
    for begin in range(0, len(calendar), CANONICAL_SESSIONS):
        check_canceled()
        last = calendar[min(begin+CANONICAL_SESSIONS, len(calendar))-1]
        warm = halos[begin]
        proofs = []
        for parent, value in direct.items():
            own = scoped_domain(value.domain, warm, last)
            token = None if input_proof is None else input_proof(parent, value, warm, last)
            if token is None and require_interval_proof:
                raise ValueError("Incremental update needs interval evidence for completed inputs; "
                    "run explicit owner index maintenance or supply input_proof before updating")
            # Initialization may bind the full durable source token. Completed
            # unchanged definitions require scoped proof, never a cold fallback.
            proof = hash_mapping({"version": FINITE_EXECUTION_POLICY,
                "content": token or {"source": value.identity, "start": str(warm), "end": str(last)}, "domain": own.signature,
                "type": value.value_type.value, "traces": value.trace_columns})
            proofs.append(proof)
        output_domain = scoped_domain(domain, calendar[begin], last)
        block_key = MaterializationKey(node_id, key.implementation_id, tuple(proofs), output_domain.signature,
            contexts(str(calendar[begin]), str(last))[0])
        record = None
        for context in contexts(str(calendar[begin]), str(last)):
            candidate_key = MaterializationKey(node_id, key.implementation_id, tuple(proofs), output_domain.signature, context)
            record = store.lookup(candidate_key, domain=output_domain, evidence=False)
            if record is not None:
                break
        if record is None:
            prefix = None
            output_start = calendar[begin]
            for candidate in sorted(partials.get(str(output_start), ()), key=lambda item: item["through"], reverse=True):
                prior_end = date.fromisoformat(candidate["through"])
                if prior_end >= last or prior_end not in calendar[begin:begin+CANONICAL_SESSIONS]:
                    continue
                prior_proofs = []
                for parent, value in direct.items():
                    own = scoped_domain(value.domain, warm, prior_end)
                    token = None if input_proof is None else input_proof(parent, value, warm, prior_end)
                    if token is None and require_interval_proof:
                        raise ValueError("Incremental prefix reuse needs explicit interval-index maintenance")
                    prior_proofs.append(hash_mapping({"version": FINITE_EXECUTION_POLICY,
                        "content": token or {"source": value.identity, "start": str(warm), "end": str(prior_end)},
                        "domain": own.signature, "type": value.value_type.value, "traces": value.trace_columns}))
                prior_domain = scoped_domain(domain, output_start, prior_end)
                prior_keys = [MaterializationKey(node_id, key.implementation_id, tuple(prior_proofs), prior_domain.signature,
                    context) for context in contexts(str(output_start), str(prior_end))]
                prior_key = next((value for value in prior_keys if value.identity == candidate["identity"]), None)
                if prior_key is not None:
                    prefix = store.lookup(prior_key, domain=prior_domain, evidence=False)
                    output_start = calendar[calendar.index(prior_end)+1]
                    break
            if policy["history_observations"] == 0:
                numerical_start = output_start
            elif node.operator in _EXACT_WINDOW_OPERATORS:
                numerical_start = _numerical_read_start(domain, output_start, last, policy["history_observations"])
            else:
                numerical_start = warm
            scoped = {}
            for (parent, value), proof in zip(direct.items(), proofs, strict=True):
                if input_window_loader is not None:
                    value = input_window_loader(parent, value, numerical_start, last)
                frame = value.lazy(include_traces=True).filter(pl.col("time").is_between(numerical_start, last)).collect().sort("time", "asset_id")
                own = scoped_domain(value.domain, numerical_start, last)
                scoped[parent] = Node.from_domain(frame, own, identity=proof, value_type=value.value_type,
                    trace_columns=value.trace_columns, trace_identity=proof if value.trace_columns else None)
                input_sessions += len(own.times)
            bindings = {declared[parent].input_key: value for parent, value in scoped.items() if declared[parent].node_type == "input"}
            retained = {parent: value for parent, value in scoped.items() if declared[parent].node_type == "operator"}
            bound = Graph.from_logical_spec(local, inputs=bindings, node_bindings=retained, parameter_bindings=parameters)
            runtime = ExecutionRuntime(check_canceled=check_canceled)
            result = runtime.run(bound, dense_output=False)
            if runtime.node_artifacts or runtime.node_checkpoints or runtime.node_training_audits:
                raise ValueError("finite contract cannot discard operator evidence")
            frame = result.collect(dense=False, include_traces=True).filter(pl.col("time") >= output_start)
            missing_domain = scoped_domain(domain, output_start, last)
            fragment_key = block_key if prefix is None else MaterializationKey(node_id, key.implementation_id,
                tuple(proofs), missing_domain.signature, hash_mapping({"fragment_of": block_key.identity, "start": str(output_start)}))
            value = Node.from_domain(frame, missing_domain, value_type=result.value_type,
                identity=fragment_key.identity, trace_columns=result.trace_columns,
                trace_identity=materialization_trace_identity(fragment_key, result.trace_columns))
            check_canceled()
            store.publish(NodeMaterialization(fragment_key, value), execution_policy=publication_policy)
            if prefix is not None:
                combined = _reference_node(pl.concat([prefix.panel.lazy(include_traces=True), value.lazy(include_traces=True)]),
                    output_domain, block_key, value_type=value.value_type, trace_columns=value.trace_columns)
                store.publish(NodeMaterialization(block_key, combined), execution_policy=publication_policy,
                    value_references=(prefix.key.identity, fragment_key.identity))
            record = store.lookup(block_key, domain=output_domain, evidence=False)
            computed += 1
            computed_sessions += len(missing_domain.times)
            kernel_sessions += len(scoped_domain(domain, numerical_start, last).times)
        else:
            reused += 1
        blocks.append(record.panel)
        references.append(record.key.identity)
    result = _reference_node(pl.concat([value.lazy(include_traces=True) for value in blocks]), domain,
        key, value_type=blocks[0].value_type, trace_columns=blocks[0].trace_columns)
    check_canceled()
    store.publish(NodeMaterialization(key, result), execution_policy=publication_policy, value_references=references)
    return store.lookup(key, domain=domain, evidence=False), {"computed_blocks": computed, "reused_blocks": reused,
        "computed_output_sessions": computed_sessions, "kernel_sessions": kernel_sessions,
        "input_sessions": input_sessions, "canonical_sessions": CANONICAL_SESSIONS}


def execute_checkpoint_node(spec, node_id, *, inputs, completed, parameters, key, domain, store, check_canceled, input_proof=None,
                            completed_materializations=(), load_inputs=None, input_window_loader=None, input_scope_proof=None):
    """Restore only when every direct parent's complete prefix is proven from records.

    The full observation calendar remains available to training windows and
    trace propagation. A checkpoint skips completed numerical state transitions;
    it never changes the frozen information cutoff or discards historical proof.
    """
    from .operator_state import capture_operator_checkpoints, operator_checkpoint_signature
    declared = {node.node_id: node for node in spec.nodes}
    relevant_parameters = _relevant_parameters(declared[node_id], parameters)
    local = LogicalGraphSpec({node_id: node_id}, spec.nodes)
    bound = Graph.from_logical_spec(local, inputs=inputs, node_bindings=completed, parameter_bindings=parameters)
    config = dict(bound._single_output().config())
    config.pop("operator")
    signature = operator_checkpoint_signature(OPERATOR_REGISTRY.get(declared[node_id].operator), config)
    registered = OPERATOR_REGISTRY.get(declared[node_id].operator)
    direct = {parent: (inputs[declared[parent].input_key] if declared[parent].node_type == "input" else completed[parent])
              for parent in parents(declared[node_id])}
    if registered.contract.checkpoint_periods is not None:
        return _execute_period_checkpoint(local, declared[node_id], registered, direct=direct,
            parameters=parameters, key=key, domain=domain, store=store,
            check_canceled=check_canceled, input_proof=input_proof,
            completed_materializations=completed_materializations, signature=signature,
            input_window_loader=input_window_loader, load_inputs=load_inputs, inputs=inputs, completed=completed)
    calendar_role = registered.contract.checkpoint_calendar_role
    anchor_calendar = ()
    if calendar_role is not None:
        calendar_parents = declared[node_id].panel_parameters.get(calendar_role, ())
        if len(calendar_parents) != 1:
            raise ValueError("checkpoint calendar role requires one explicit auxiliary Node")
        anchor_calendar = tuple(direct[calendar_parents[0]].domain.times.to_list())
        actual_coverage = tuple(day for day in anchor_calendar
            if domain.times.min() <= day <= domain.times.max())
        if actual_coverage != tuple(domain.times.to_list()):
            raise ValueError("explicit checkpoint calendar axis differs from actual execution calendar")
    def parent_scopes(end):
        return _checkpoint_parent_scopes(declared[node_id], registered.contract, config, domain,
            direct, end, scoped=input_scope_proof is not None)
    def proofs(end, ends=None, cutoffs=None):
        result = {}
        for parent, value in direct.items():
            bound = end if ends is None else ends[parent]
            if bound is None:
                result[parent] = hash_mapping({"empty_consumed_prefix": True, "type": value.value_type.value})
                continue
            bound = date.fromisoformat(str(bound))
            if cutoffs and parent in cutoffs:
                token = input_scope_proof(parent, value, domain.times.min(), bound, date.fromisoformat(cutoffs[parent]))
            else:
                token = None if input_proof is None else input_proof(parent, value, domain.times.min(), bound)
            if token is None:
                return None
            result[parent] = token
        return result
    ends, cutoffs = parent_scopes(domain.times.max())
    policy = {"name": "checkpoint_prefix.v2", "parameters": relevant_parameters,
        "parent_ends": ends, "parent_proofs": proofs(domain.times.max(), ends, cutoffs)}
    if calendar_role is not None:
        policy["anchor_calendar"] = [str(day) for day in anchor_calendar if day <= domain.times.max()]
        policy["calendar_role"] = calendar_role
    if cutoffs:
        policy["parent_cutoffs"] = cutoffs
    previous = None
    manifests = _checkpoint_candidates(store, declared[node_id], key, domain, parameters, parent_scopes,
        calendar_role=calendar_role, anchor_calendar=anchor_calendar)
    for manifest in manifests:
        check_canceled()
        end = date.fromisoformat(manifest["through"])
        saved_cutoffs = manifest["execution_policy"].get("parent_cutoffs")
        current_proofs = proofs(end, manifest["execution_policy"].get("parent_ends"), saved_cutoffs)
        if current_proofs is None or manifest["execution_policy"]["parent_proofs"] != current_proofs:
            continue
        prior_domain = None
        if calendar_role is not None:
            prior_domain = scoped_domain(domain, domain.times.min(), end,
                                         identity=manifest["key"]["domain_identity"])
            matching_domain = store.domain_matches(manifest["identity"], prior_domain)
            if matching_domain is None:
                raise ValueError("Completed checkpoint Domain needs explicit Core interval-index maintenance")
            if not matching_domain:
                continue
        try:
            candidate = (store.read(manifest["identity"]) if calendar_role is None else
                replace(store.lookup(MaterializationKey.from_dict(manifest["key"]), domain=prior_domain, evidence=False),
                    checkpoint=store.read_checkpoint(manifest["identity"])))
            if scoped_domain(domain, domain.times.min(), end).signature != scoped_domain(
                    candidate.panel.domain, domain.times.min(), end).signature:
                continue
            checkpoint = candidate.checkpoint
            if (not isinstance(checkpoint, Mapping) or checkpoint.get("signature") != signature
                    or not isinstance(checkpoint.get("state"), Mapping)
                    or checkpoint["state"].get("through") != str(end)
                    or (registered.contract.checkpoint_validator is not None
                        and not registered.contract.checkpoint_validator(checkpoint["state"]))):
                continue
        except (ValueError, OSError, pl.exceptions.PolarsError):
            # A damaged optional checkpoint does not invalidate new source evidence.
            continue
        previous = candidate
        break
    if previous is None:
        for identity in completed_materializations:
            manifest = store.manifest(identity)
            prior_policy = manifest.get("execution_policy") or {}
            if (manifest["key"]["implementation_id"] == key.implementation_id
                    and ("parameters" not in prior_policy
                        or _relevant_parameters(declared[node_id], prior_policy["parameters"]) == relevant_parameters)
                    and manifest["coverage_start"] == str(domain.times.min())):
                raise ValueError("Completed checkpoint history has no proven preceding state; "
                    "plan affected intervals or run explicit checkpoint/input-proof maintenance")
    if previous is not None and previous.checkpoint["state"]["through"] == str(domain.times.max()):
        retained_artifacts = previous.artifacts if calendar_role is None else {
            name: store.compose_artifact(previous.key.identity, name)
            for name in store.manifest(previous.key.identity)["artifacts"]}
        value = _reference_node(previous.panel.lazy(include_traces=True), domain, key,
            value_type=previous.panel.value_type, trace_columns=previous.panel.trace_columns)
        store.publish(NodeMaterialization(key, value, artifacts=retained_artifacts, checkpoint=previous.checkpoint,
            training_audits=previous.training_audits), execution_policy=policy, value_references=(previous.key.identity,))
        return store.lookup(key, domain=domain, evidence=False), {
            "checkpoint_through": str(domain.times.max()), "replayed_sessions": 0}
    registered = OPERATOR_REGISTRY.get(declared[node_id].operator)
    history = registered.contract.checkpoint_history
    through = None if previous is None else date.fromisoformat(previous.checkpoint["state"]["through"])
    if through is not None and history is not None and input_window_loader is None and load_inputs is not None:
        load_inputs()
        direct = {parent: (inputs[declared[parent].input_key] if declared[parent].node_type == "input" else completed[parent])
                  for parent in parents(declared[node_id])}
    if through is not None and history is not None:
        observations = history(config) if callable(history) else history
        if isinstance(observations, bool) or not isinstance(observations, int) or observations < 0:
            raise ValueError("checkpoint history must be a nonnegative observation count")
        calendar = domain.times.to_list()
        begin = max(0, calendar.index(through) + 1 - observations)
        warm, last = calendar[begin], calendar[-1]
        scoped = {}
        for parent, value in direct.items():
            if input_window_loader is not None:
                value = input_window_loader(parent, value, warm, last)
            own = scoped_domain(value.domain, warm, last)
            frame = value.lazy(include_traces=True).filter(pl.col("time").is_between(warm, last))
            scoped[parent] = Node.from_domain(frame, own, identity=value.identity, value_type=value.value_type,
                trace_columns=value.trace_columns, trace_identity=value.trace_identity)
        bindings = {declared[parent].input_key: value for parent, value in scoped.items() if declared[parent].node_type == "input"}
        retained = {parent: value for parent, value in scoped.items() if declared[parent].node_type == "operator"}
        bound = Graph.from_logical_spec(local, inputs=bindings, node_bindings=retained, parameter_bindings=parameters)
    elif load_inputs is not None:
        load_inputs()
        bound = Graph.from_logical_spec(local, inputs=inputs, node_bindings=completed, parameter_bindings=parameters)
    runtime = ExecutionRuntime(check_canceled=check_canceled)
    restored = {} if previous is None else {node_id: previous.checkpoint}
    with capture_operator_checkpoints(restored, calendar=domain.times.to_list(),
            anchor_calendars={node_id: anchor_calendar} if calendar_role is not None else {}) as checkpoints:
        result = runtime.run(bound, dense_output=False)
        frame = result.collect(dense=False, include_traces=True)
    artifacts = dict(runtime.node_artifacts.get(node_id, {}))
    audits = tuple(runtime.node_training_audits.get(node_id, ()))
    checkpoint = checkpoints.captured.get(node_id)
    from .operation_contract import TraceRule
    if registered.contract.trace_rule == TraceRule.CUMULATIVE_MAX and result.trace_columns:
        carry = None if previous is None else previous.checkpoint.get("trace_state")
        if previous is not None and carry is None:
            # Original older checkpoints lack this auxiliary trace carry. Value
            # consumption may derive it once; readiness never reads value bytes.
            carry = previous.panel.lazy(include_traces=True).group_by("asset_id").agg(
                *[pl.col(name).max() for name in result.trace_columns]).collect().to_dicts()
        if carry:
            names = {name: f"_previous_{name}" for name in result.trace_columns}
            prior = pl.DataFrame(carry, schema={"asset_id": pl.String,
                **{name: frame.schema[name] for name in result.trace_columns}}).rename(names)
            frame = frame.join(prior, on="asset_id", how="left").with_columns(
                *[pl.max_horizontal(name, names[name]).alias(name) for name in names]).drop(*names.values())
        if checkpoint is not None:
            tail_carry = frame.group_by("asset_id").agg(*[pl.col(name).max() for name in result.trace_columns])
            if carry:
                tail_carry = pl.concat([pl.DataFrame(carry, schema=tail_carry.schema), tail_carry]).group_by("asset_id").agg(
                    *[pl.col(name).max() for name in result.trace_columns])
            checkpoint = {**checkpoint, "trace_state": tail_carry.sort("asset_id").to_dicts()}
    if previous is not None:
        frame = frame.filter(pl.col("time") > through)
        audits = (*previous.training_audits, *audits)
        if calendar_role is not None:
            for name in store.manifest(previous.key.identity)["artifacts"]:
                artifacts[name] = store.compose_artifact(previous.key.identity, name, artifacts.get(name))
        for name in (() if calendar_role is not None else set(previous.artifacts) | set(artifacts)):
            old, new = previous.artifacts.get(name), artifacts.get(name)
            if isinstance(old, pl.DataFrame) and isinstance(new, pl.DataFrame) and "time" in old.columns:
                artifacts[name] = pl.concat([old.filter(pl.col("time") <= through), new.filter(pl.col("time") > through)]).sort("time")
            elif name not in artifacts:
                artifacts[name] = old
    references = None
    if previous is not None:
        tail_domain = domain.window(through + timedelta(days=1), domain.times.max())
        tail_key = MaterializationKey(node_id, key.implementation_id, key.inputs, tail_domain.signature,
            hash_mapping({"continuation_of": key.identity, "after": str(through)}))
        tail = Node.from_domain(frame, tail_domain, identity=tail_key.identity, value_type=result.value_type,
            trace_columns=result.trace_columns, trace_identity=materialization_trace_identity(tail_key, result.trace_columns))
        check_canceled()
        store.publish(NodeMaterialization(tail_key, tail), execution_policy=policy)
        references = (previous.key.identity, tail_key.identity)
        frame = pl.concat([previous.panel.lazy(include_traces=True), tail.lazy(include_traces=True)])
    value = (_reference_node(frame, domain, key, value_type=result.value_type, trace_columns=result.trace_columns)
        if references is not None else Node.from_domain(frame, domain, identity=key.identity, value_type=result.value_type,
            trace_columns=result.trace_columns, trace_identity=materialization_trace_identity(key, result.trace_columns)))
    record = NodeMaterialization(key, value, artifacts=artifacts, checkpoint=checkpoint, training_audits=audits)
    check_canceled()
    store.publish(record, execution_policy=policy, value_references=references)
    return (store.lookup(key, domain=domain, evidence=False) if calendar_role is not None else store.read(key.identity)), {"checkpoint_through": None if through is None else str(through),
        "replayed_sessions": len(domain.times) if through is None else len(domain.times.filter(domain.times > through))}


def _period_runs(selected, calendar):
    """Ordered ranges containing exactly the selected dates of this axis."""
    result, first, previous = [], None, None
    selected = set(selected)
    for day in calendar:
        if day in selected:
            if first is None:
                first = day
            previous = day
        elif first is not None:
            result.append((first, previous))
            first = None
    if first is not None:
        result.append((first, previous))
    return result


def _period_domain(domain, dates):
    dates = pl.Series("time", dates, dtype=pl.Date)
    universe = (domain.grid_lazy().filter(pl.col("time").is_in(dates.implode())).collect()
        .with_columns(pl.lit(True).alias("active")) if domain.is_dynamic else domain.asset_ids)
    return Domain(calendar=dates, universe=universe)


def _execute_period_checkpoint(local, node, registered, *, direct, parameters, key, domain, store,
        check_canceled, input_proof, completed_materializations, signature, input_window_loader,
        load_inputs, inputs, completed):
    """Run independent statistics and their affected predictions in one runtime.

    Source proofs, state validation and bounded read admission precede kernels.
    The numerical owner selects maturity work and records exact output dates;
    this planner owns caches, immutable references, cancellation and publication.
    """
    from .operator_state import capture_operator_checkpoints
    calendar = tuple(domain.times.to_list())
    if not calendar:
        raise ValueError("period checkpoint requires an explicit nonempty calendar")
    relevant = _relevant_parameters(node, parameters)
    roles = {parent: tuple(((["features"] if parent in node.inputs else []) +
        [role for role, values in node.panel_parameters.items() if parent in values])) for parent in direct}
    period_proofs = {}
    for day in calendar:
        check_canceled()
        evidence = {parent: (None if input_proof is None else input_proof(parent, value, day, day))
                    for parent, value in direct.items()}
        if any(not isinstance(token, str) or not token for token in evidence.values()):
            raise ValueError(f"Prediction period {day} requires explicit source interval-proof maintenance")
        period_proofs[str(day)] = evidence
    policy = {"name": "checkpoint_periods.v1", "role": "complete", "parameters": relevant, "period_proofs": period_proofs}
    previous, restored = None, None
    candidates = []
    for item in store.materialization_candidates(node.node_id, key.implementation_id,
            policy_name=policy["name"], parameter_key=hash_mapping(relevant), policy_role="complete",
            coverage_start=str(calendar[0]), through=str(calendar[-1]), checkpoint=True):
        if candidates and item["through"] < candidates[0][0]["through"]:
            break  # Only the newest valid boundary participates in tie-breaking.
        manifest = store.manifest(item["identity"])
        saved_policy = manifest.get("execution_policy") or {}
        if (saved_policy.get("name") != policy["name"] or saved_policy.get("role") != "complete"
                or saved_policy.get("parameters") != relevant
                or manifest["coverage_start"] != str(calendar[0]) or manifest["through"] > str(calendar[-1])):
            continue
        check_canceled()
        try:
            checkpoint = store.read_checkpoint(item["identity"])
            state = checkpoint["state"]
            proofs = saved_policy["period_proofs"]
            if (checkpoint["signature"] != signature or not registered.contract.checkpoint_validator(state)
                    or state["through"] != manifest["through"] or state["calendar"][0] != manifest["coverage_start"]
                    or set(proofs) != set(state["calendar"])
                    or any(set(value) != set(direct) or any(not isinstance(token, str) or not token
                        for token in value.values()) for value in proofs.values())):
                continue
            differences = sum(proofs.get(day) != current for day, current in period_proofs.items()
                              if day <= manifest["through"])
            candidates.append((manifest, checkpoint, differences))
        except (KeyError, TypeError, ValueError, OSError, pl.exceptions.PolarsError):
            continue
    if candidates:
        candidates.sort(key=lambda item: (item[0]["through"], -item[2]), reverse=True)
        previous, restored, _ = candidates[0]
    if previous is None:
        for identity in completed_materializations:
            manifest = store.manifest(identity)
            saved_policy = manifest.get("execution_policy") or {}
            saved_parameters = saved_policy.get("parameters", saved_policy.get("parameter_bindings"))
            if (manifest["key"]["implementation_id"] == key.implementation_id
                    and (saved_parameters is None or _relevant_parameters(node, saved_parameters) == relevant)):
                raise ValueError("Completed Prediction has no original per-period statistical evidence; "
                    "run explicit evidence maintenance from saved IC/regression records, never refit old periods")
    old_proofs = {} if previous is None else previous["execution_policy"]["period_proofs"]
    changes = {day: tuple(sorted({role for parent, proof in period_proofs[str(day)].items()
        if old_proofs.get(str(day), {}).get(parent) != proof for role in roles[parent]})) for day in calendar
        if old_proofs.get(str(day)) != period_proofs[str(day)]}
    old_state = None if restored is None else restored["state"]
    if old_state is not None:
        consumed = {period for history in old_state["fit_windows"].values() for period in (history or [])}
        revised = {str(day) for day in changes if str(day) in old_proofs}
        outputs = {day for day, history in old_state["fit_windows"].items()
                   if history is not None and old_state["records"][day]["has_features"]}
        if outputs and (consumed and consumed <= revised or all("features" in changes.get(date.fromisoformat(day), ()) for day in outputs)):
            raise ValueError("Prediction change affects all completed prediction history; plan explicit affected intervals before repair")
    source_periods = registered.contract.checkpoint_periods(old_state, calendar, changes)
    if (source_periods != tuple(day for day in calendar if day in set(source_periods))):
        raise ValueError("period checkpoint owner selected an invalid or unordered source interval")
    if previous is not None and input_window_loader is None and load_inputs is not None and source_periods:
        raise ValueError("Completed Prediction requires a bounded input reader before updating period statistics")
    declared = {entry.node_id: entry for entry in local.nodes}
    read_periods = set()

    def run_phase(dates, checkpoint, phase):
        scoped = {}
        for parent, value in direct.items():
            parts = []
            for first, last in _period_runs(dates, calendar):
                check_canceled()
                selected = value if input_window_loader is None else input_window_loader(parent, value, first, last)
                parts.append(selected.lazy(include_traces=True).filter(pl.col("time").is_between(first, last)))
            own = _period_domain(value.domain, dates or (calendar[0],))
            frame = pl.concat(parts) if parts else value.lazy(include_traces=True).filter(pl.lit(False))
            scoped[parent] = Node.from_domain(frame, own, identity=value.identity, value_type=value.value_type,
                trace_columns=value.trace_columns, trace_identity=value.trace_identity)
        bindings = {declared[parent].input_key: value for parent, value in scoped.items() if declared[parent].node_type == "input"}
        retained = {parent: value for parent, value in scoped.items() if declared[parent].node_type == "operator"}
        bound = Graph.from_logical_spec(local, inputs=bindings, node_bindings=retained, parameter_bindings=parameters)
        runtime = ExecutionRuntime(check_canceled=check_canceled)
        work = {"phase": phase, "input_periods": [str(day) for day in dates],
                "changes": {str(day): list(values) for day, values in changes.items()}}
        with capture_operator_checkpoints({} if checkpoint is None else {node.node_id: checkpoint},
                calendar=calendar, period_work={node.node_id: work}) as captured:
            result = runtime.run(bound, dense_output=False)
            frame = result.collect(dense=False, include_traces=True)
        current = captured.captured.get(node.node_id)
        if (current is None or current["signature"] != signature
                or not registered.contract.checkpoint_validator(current["state"])
                or current["state"]["calendar"] != [str(day) for day in calendar]):
            raise ValueError("period checkpoint did not publish complete original statistical evidence")
        read_periods.update(dates)
        return result, frame, current

    partial_key = MaterializationKey(node.node_id, key.implementation_id, key.inputs, domain.signature,
        hash_mapping({"period_statistics_for": key.identity, "previous": None if previous is None else previous["identity"]}))
    cached = store.manifest(partial_key.identity)
    checkpoint = None
    statistics_policy = {**policy, "role": "statistics"}
    if cached is not None and cached.get("execution_policy") == statistics_policy:
        candidate = store.read_checkpoint(partial_key.identity)
        if (candidate["signature"] == signature and registered.contract.checkpoint_validator(candidate["state"])
                and candidate["state"]["calendar"] == [str(day) for day in calendar]
                and "repaired_output_periods" in candidate["state"]):
            checkpoint = candidate
    if checkpoint is None and (source_periods or old_state is not None and old_state["calendar"] != [str(day) for day in calendar]):
        if previous is None and load_inputs is not None and input_window_loader is None:
            load_inputs()
            direct = {parent: (inputs[declared[parent].input_key] if declared[parent].node_type == "input" else completed[parent])
                      for parent in parents(node)}
        result, _, checkpoint = run_phase(source_periods, restored, "statistics")
        empty = Node.from_domain(result.lazy(include_traces=True).filter(pl.lit(False)), domain,
            value_type=result.value_type, trace_columns=result.trace_columns,
            trace_identity=materialization_trace_identity(partial_key, result.trace_columns))
        check_canceled()
        store.publish(NodeMaterialization(partial_key, empty, checkpoint=checkpoint), execution_policy=statistics_policy)
    elif checkpoint is None:
        checkpoint = {**restored, "state": {**old_state, "repaired_output_periods": []}}
    output_periods = tuple(date.fromisoformat(day) for day in checkpoint["state"]["repaired_output_periods"])
    if output_periods != tuple(day for day in calendar if day in set(output_periods)):
        raise ValueError("period checkpoint owner selected invalid output coverage")
    references = []
    delta_identity = None
    if output_periods:
        output_domain = _period_domain(domain, output_periods)
        delta_key = MaterializationKey(node.node_id, key.implementation_id, key.inputs, output_domain.signature,
            hash_mapping({"period_predictions_for": key.identity, "statistics": partial_key.identity}))
        cached_delta = store.manifest(delta_key.identity)
        delta_policy = {**policy, "role": "prediction_delta"}
        if cached_delta is not None and cached_delta.get("execution_policy") == delta_policy:
            result = store.lookup(delta_key, domain=output_domain, evidence=False).panel
            checkpoint = store.read_checkpoint(delta_key.identity)
            if (checkpoint["signature"] != signature or not registered.contract.checkpoint_validator(checkpoint["state"])
                    or checkpoint["state"]["calendar"] != [str(day) for day in calendar]):
                raise ValueError("cached Prediction delta lost original statistical evidence")
        else:
            result, frame, checkpoint = run_phase(output_periods, checkpoint, "prediction")
            delta = Node.from_domain(frame, output_domain, value_type=result.value_type,
                trace_columns=result.trace_columns, trace_identity=materialization_trace_identity(delta_key, result.trace_columns))
            check_canceled()
            store.publish(NodeMaterialization(delta_key, delta, checkpoint=checkpoint), execution_policy=delta_policy)
        delta_identity = delta_key.identity
    else:
        checkpoint = {**checkpoint, "state": {name: value for name, value in checkpoint["state"].items()
                                              if name != "repaired_output_periods"}}
        source_key = MaterializationKey.from_dict(previous["key"])
        result = store.lookup(source_key, domain=scoped_domain(domain, calendar[0], date.fromisoformat(previous["through"]),
            identity=source_key.domain_identity), evidence=False).panel
    old_calendar = () if old_state is None else tuple(date.fromisoformat(day) for day in old_state["calendar"])
    union = sorted(set(calendar) | set(old_calendar))
    current = set(calendar)
    changed_outputs = set(output_periods)
    origin, first, last = None, None, None
    for day in union:
        selected_origin = (None if day not in current else delta_identity if day in changed_outputs else
                           None if previous is None else previous["identity"])
        if selected_origin != origin:
            if origin is not None:
                references.append({"identity": origin, "start": first, "end": last})
            origin, first = selected_origin, day
        last = day
    if origin is not None:
        references.append({"identity": origin, "start": first, "end": last})
    value = _reference_node(result.lazy(include_traces=True), domain, key, value_type=result.value_type,
                            trace_columns=result.trace_columns)
    check_canceled()
    store.publish(NodeMaterialization(key, value, checkpoint=checkpoint), execution_policy=policy, value_references=references)
    return store.lookup(key, domain=domain, evidence=False), {
        "checkpoint_through": None if previous is None else previous["through"],
        "replayed_sessions": len(output_periods), "statistic_periods": len(source_periods) if cached is None else 0,
        "output_periods": [str(day) for day in output_periods], "input_periods": [str(day) for day in sorted(read_periods)]}
