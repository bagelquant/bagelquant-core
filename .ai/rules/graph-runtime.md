# Logical graph, materialization and resources

## Target ownership and current implementation

- Core's target owns generic numerical artifacts, materialization/cache stores,
  checkpoints, their integrity/reuse and atomic publication through public APIs.
  The current MaterializationStore is a protocol; generic application-backed
  implementations and planning mechanisms in Workbench remain to move in stage 3.
  This rule change does not implement a store or define future APIs/schemas.
- Data owns frozen input datasets/evidence; Core consumes explicit neutral inputs
  and validates numerical reuse against their immutable identities. Core never
  imports or queries Data, BT or Workbench. BT owns account/evaluation storage.
- Workbench freezes research intent, China semantics and backend receipt references
  and orchestrates tasks; it does not own generic numerical storage or cache logic.

## Identity and execution

- Immutable content-addressed `LogicalGraphSpec` is the sole persistent
  calculation graph; authoring `GraphSpec` templates compile into it. Raw stays
  outside Core. Node/Operator registration and DAG JSON/Mermaid belong here.
- Node IDs include canonical operations, expanded parameters, ordered inputs,
  named auxiliary roles and type contracts. Author aliases, object IDs, stage,
  Snapshot, input bytes and hardware budgets are logical-identity neutral.
- Materialization keys include implementation/kernel versions, relevant frozen
  input identities, Domain and numerical context. Unrelated source updates must
  not invalidate the whole DAG. Typed values, traces, fit audits, checkpoints
  and decisions share one immutable result receipt.
- Explicit projection/applicability and 14-significant-digit Alpha/Prediction
  boundaries are ordinary generic operations. Never round intermediate results.
- Numerical operators perform no provider/governance or hidden storage I/O.
  The target Core storage/runtime boundary owns numerical artifact publication,
  retention and invalidation; Data owns source/frozen-input authority. Workbench
  owns research metadata and governance. Supplied input identities must prove
  immutable content; current application-backed storage remains a migration gap.

## Causal history and cache planning

- Expose causal observation-history bounds and materialization-key planning.
  Exact-key planning must not execute operators or collect numerical payloads;
  exact cache hits do not load numerical source payloads.
- Observation counts follow each asset's admitted coordinates; sparse active
  membership needs a proven warm-up/halo. Unknown/unbounded history or an
  unproven incremental prefix requires conservative complete-history computation.
  Causal bounds alone never prove checkpoint/source-prefix integrity.
- Numerical execution policies participate in selected node context and
  propagate through parent keys. Hardware and physical chunks remain neutral.
  Do not reuse full-history prefixes as canonical finite-block receipts or
  implicitly treat block receipts as full-domain results. Current Workbench uses
  fixed 32-session `finite_causal_blocks.v1` and `dag.values.v4` bindings; generic
  block/reuse/publication mechanisms are pending migration to Core in stage 3,
  while Workbench retains research binding metadata and task orchestration.
- Keep observation-window endpoints distinct from the frozen information
  cutoff. Canonical prefix proofs use stable row order rather than scan order.

## Resource ownership

- Share one total thread/memory budget across workers and native threads.
  Hardware limits are identity-neutral; sample/model/solver settings are
  numerical operator parameters. Never truncate required mathematical history
  to satisfy an operational budget.
- Admit independent eager kernels through a bounded ready frontier; preserve
  fused lazy plans. Merge worker evidence in stable order on the coordinator
  thread. Native pools and LightGBM thread/histogram allocations share the
  concurrent worker budget; audit/checkpoint contexts stay isolated.
- Memory pressure shrinks later batches, defers new nodes and releases optional
  caches. Record observed peak RSS, duration, pressure and actual batches/threads;
  the memory target remains soft. Bound caller date blocks while preserving
  numerical history and fixed arithmetic tile boundaries.
