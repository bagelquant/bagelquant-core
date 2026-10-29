# Public API

Use `Domain`, `Node`, `ValueType`, `Operator`, `Graph`, `CoreStore`,
`ExecutionRuntime`, `MaterializationKey` and `NodeMaterialization` from
`bagelquant_core`. Registered numerical operations live in
`bagelquant_core.operator`.

The [architecture guide](../architecture.md) documents persistent execution and
types; the [catalog](operators/index.md) documents each operator.

- `ArtifactVerification` provides finite, bounded checksum proofs keyed by receipt and file identity. `CoreStore.read_context()` owns its lifetime.
- `CoreStore.describe_update(identity)` and `describe_evidence(identity)` inspect committed metadata without loading numerical artifacts.
- `Graph.plan_update(..., persist_nodes=(), native_admission=None, kernel_dispatch=None)` preserves canonical mathematics while choosing durable boundaries and caller-admitted execution. `plan.regions` describes bounded pointwise regions; existing `ready`/`execute` calls use durable boundary IDs. `plan.logical_coverage` names satisfied transient nodes separately from real artifact results; `plan.preparation_usage` reports bounded preparation and transport work.
- `CoreStore.update_covers(receipt_id, context_id, node_ids=...)` validates registered receipt/seal authority before checking logical coverage. Receipts include `coverage[context][transient_node] = durable_boundary_node`; a transient node has no fabricated materialization ID.
- `UpdatePlan.required_memory_bytes(context_id, node_id, limits=...)` sizes missing finite/region work from registered metadata and resident Domains. It returns zero for complete work and `None` for unknown sizing. Callers choose concurrency within their existing budgets; execution still validates original proofs and local limits.
- `FiniteKernelWork`/`FiniteKernelResult` support immutable typed, JSON-compatible staging descriptors. Work carries a required positive `estimated_payload_bytes` covering inputs and every retained output. `compute_finite_work(work)` checks that estimate against its local limits before staged reads, then performs only numerical work, including required transient proof frames; the coordinator validates, publishes and advances completion. Workbench owns fresh process setup and admission. A dispatcher may supply `should_dispatch(estimated_payload_bytes)` to retain small work inline before Arrow staging. Missing finite/region intervals fail before source reads when the conservative working estimate exceeds their admitted memory; completed history remains reusable.
- `CoreStore.publication_context()` amortizes explicit initialization for a finite write unit. Writer transactions still check schema and publication authority; replacing the metadata file invalidates the context.
- `CoreStore.read_values(identity, start=..., end=..., include_traces=...)` reads selected calendar/membership observations and monthly values, returning typed sparse values without byte rehashing. Empty windows preserve types/traces; unbounded reads consume the complete Domain. `verify_update` and `check_integrity` remain full audits.
- `CoreStore.publish(..., value_references=[{"identity": ..., "start": ..., "end": ...}, ...])` composes original typed intervals and new repair materializations. References must cover the complete ordered calendar exactly once with identical original membership/type/traces. Each retained Parquet partition carries its selected bounds, which are pushed down on reads. Publication fences original authority and file existence without historical decoding or rewriting.
- `CoreStore.describe_artifact(identity, name)` describes the same recursively selected original channel as `evidence`; `artifact_interval_identity(identity, name, start=..., end=...)` proves all dated columns, including decision status/reason and empty calendar dates, from original-bound metadata. Missing optional proof returns `None` without reading values.
- `CoreStore.read_artifact(identity, name, start=..., end=...)` prunes original concat leaves before Parquet reads and pushes date predicates down. `artifact_index_plan(identity, name)` / `build_artifact_index(plan)` explicitly audit and index older original artifacts while preserving receipts/files. Missing legacy indexes require this explicit maintenance, never an implicit historical read. Subsequent concatenations reuse the retained original index.


`CoreStore.value_read_rows(identity, start=..., end=...)` returns a metadata-only
conservative row bound for a bounded value read: overlapping physical partitions,
selected calendar/Domain coordinates and consumed membership. It returns `None`
for missing optional counts. Callers choose memory policy; Core neither detects
hardware nor grants global admission. Explicit `build_index(index_plan())` enriches
old counts without changing receipt/content/interval identities. Passive reads
never scan numerical bodies or initialize/rebuild these indexes.
