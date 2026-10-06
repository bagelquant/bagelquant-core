# Graph, cache, lifecycle and execution

Core owns LogicalGraphSpec, Graph DSL/merge/resolve, dependency primitives, and
CoreStore SQLite/Parquet implementations at caller-provided paths. It imports no
Data/BT/Workbench. Workbench owns global admission, CPU/RAM detection and pressure
policy; Core only applies explicit local budgets and reports work.

Logical identity excludes names, app IDs, Domain, source versions and hardware.
Material keys include relevant evidence, Domain, implementation/kernel and numerical
context. Source identities prove immutable values/traces; no unproven reuse.

Use fixed 32-session finite mathematical blocks with a caller anchor. Prove all
parent values, coordinates, dynamic membership and availability for block reuse.
Declared checkpoint replay requires unchanged complete parent prefixes, preserves
full calendar/training context and merges audits/artifacts. Unknown or unsafe
history falls back to required full computation. Never substitute observation
endpoints for information cutoffs. Worker counts and physical batches are neutral.

Global updates freeze graph/state revisions and every registered Domain context.
All active computed nodes must succeed before the global pointer advances. Branch
receipts do not advance it. Failures/cancellation/CAS conflicts retain cache only.
Sleep atomically cascades downstream across contexts and is checked at merge,
admission, execution and publication. Wake is explicit and atomic with active
upstream prerequisites. Cached values never bypass sleep; historical reads remain.

CoreStore owns type/Domain/trace/checkpoint/audit/artifact integrity, inventory,
recovery and cleanup. Public receipt APIs hide backend file locations from apps.
reference_publication fences Core state while an application commits an immutable
receipt reference in its own transaction; no distributed commit or rollback.
Cleanup preserves committed historical evidence and only reclaims verified
unreferenced completed generations. New storage rejects old schemas.

Run standalone temporary-store tests and owner checks before BT/Workbench.
See [architecture](../../docs/en/architecture.md) for callable examples.
