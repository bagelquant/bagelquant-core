# Graph, cache, lifecycle and execution

Retained older execution-policy receipts cannot veto independently completed
current-policy prefixes with matching relevant parameters. Reuse still requires
current block/input/parameter proofs before any kernel runs. If only obsolete
policies remain, or current consumed proofs are unknown, require explicit
maintenance rather than adopting the old policy or replaying completed history.

Core owns LogicalGraphSpec, Graph DSL/merge/resolve, dependency primitives, and
CoreStore SQLite/Parquet implementations at caller-provided paths. It imports no
Data/BT/Workbench. Workbench owns global admission, CPU/RAM detection and pressure
policy; Core only applies explicit local budgets and reports work.

Schema 3 stores one compact `node_completions` row per graph/context/logical node.
`CoreStore.node_completions` reads successful definition/upstream digests,
observation/check dates and the immutable result reference in one metadata query.
`UpdatePlan` adopts only exact matching authority without a value lookup or proof
callback. Missing children construct lazy plans for only their consumed completed
parents. Each durable result advances its flag under graph/state revision fences;
incomplete updates retain these flags while global/branch completion remains a
separate publication. Older historical branches cannot regress the latest flag.
A date alone never proves freshness after definition, Domain or dependency edits.
Update headers, admitted contexts and node-result relations are normalized and
published atomically. Indexed completion/request queries never scan enclosing
receipt JSON. Materialization candidate headers bind the original content,
implementation, policy, parameters, dates and checkpoint availability. Planners
filter these records before reading selected manifests/checkpoints; exact public
update bodies and identities are reconstructed from the registered relations.
Finite read contexts admit bounded thread-safe immutable manifest/index memos;
they expire on exit and never cache revision fences or missing optional indexes.
Explicit audits bypass metadata memos and check candidate/original bindings.
Schema 2 is incompatible; there is no migration reader or implicit reconstruction.

`plan_update(causal_inputs=..., input_version_loader=..., input_scope_proof=...,
causal_output_domains=...)` admits typed CausalInputWindow versions in the same
update engine. Effective known_at/revision precedence comes from the source;
null versions and late-first natural keys are retained. Source coverage may be
larger than the published causal coordinate union. Interval keys bind scoped
as-of proofs and historical output membership, not future source assets/keys.
Normal appends share 32-session batches; late old-observation visibility splits
only affected output ranges and reselects complete finite dependency halos.
Calendar-dense temporary windows preserve null suspension sessions. Version
buffers retain bounded halos, acquire new observation ranges once under frozen
ceilings, and never create decision×lookback×asset historical tensors. Published
node values and completion flags remain Core authority; no application-specific
Risk formula or provider/storage dependency belongs in this mechanism.

`input_definition_identities` admits stable source numerical-definition evidence
per context/input, separately from changing content and scoped as-of tokens.
When supplied it covers exactly admitted contexts and selected input keys.
Causal materialization/completion/block keys bind only relevant ancestors. A
source implementation change creates new authority without deleting original
history; an unchanged source definition never excuses unproven history. Do not
bind full frozen receipts, dynamic content/cutoffs, display text or resources as
source definition identities. Public causal policy version 2 covers this contract.

Logical identity excludes names, app IDs, Domain, source versions and hardware.
Material keys include relevant evidence, Domain, implementation/kernel and numerical
context. Source identities prove immutable values/traces; no unproven reuse.

Use fixed 32-session finite mathematical blocks with a caller anchor. Prove all
parent values, coordinates, dynamic membership and availability for block reuse.
Declared checkpoint replay requires unchanged complete parent prefixes, preserves
full calendar/training context and merges audits/artifacts. For calculation changes,
unknown/unsafe prefix evidence requires an actionable plan or explicit proof
maintenance, never silent full-history replay or unproven reuse. Never substitute
observation endpoints for information cutoffs. Workers and physical batches are neutral.

Unchanged numerical definitions update only new/affected coordinates and reuse
completed blocks/checkpoints, old fit events and causal state for portfolio targets. Keep
necessary lookback reads distinct from recomputed output. New cutoffs, enclosing
receipts/Domain extensions and unrelated global counters are not full-rebuild
reasons. Full computation is initial materialization or a relevant definition,
operator implementation or numerical-parameter change. Preserve scheduled new
fits, maturity/embargo/seed semantics and all old model evidence. Test append and
local repair against a full reference, with call/date/fit instrumentation proving
unchanged historical work does not run; record current implementation gaps.
Do not change retained canonical proof scopes or material keys merely to reduce
numerical reads. Narrow a missing output's numerical window only for kernels
with proven exact value/trace independence from earlier state. Count admitted
membership coordinates, including nulls, and preserve old observations needed
by reentering assets. Permanently exited assets absent from the missing output
must not extend such a window. Other finite rolling kernels may depend on
earlier floating-point accumulator state and retain their canonical warm read.
Old fits are reusable only with unchanged complete features/labels/samples,
availability/maturity, embargo, seed and numerical parameters. Repair affected
fit events/predictions on relevant input revision. Cross-sectional impact covers
all affected-date members; recursive state replays the affected suffix from a
valid preceding checkpoint. Proven all-history impact is an explicit blocked
plan under this policy, not an implicit full fallback or fake definition edit.
Canonical Date columns keep an identity normalization expression so observation
window predicates reach persisted Parquet scans before decompression. Real String/
Datetime conversions remain explicit; values, traces and material keys are unchanged.

Global updates freeze graph/state revisions and every registered Domain context.
All active computed nodes must succeed before the global pointer advances. Branch
receipts do not advance it. Failures/cancellation/CAS conflicts retain cache only.
Sleep atomically cascades downstream across contexts and is checked at merge,
admission, execution and publication. Wake is explicit and atomic with active
upstream prerequisites. Cached values never bypass sleep; historical reads remain.

CoreStore owns type/Domain/trace/checkpoint/audit/artifact integrity, inventory,
recovery and cleanup. Public receipt APIs hide backend file locations from apps.
`describe`, `describe_update` and `describe_evidence` validate committed metadata
without hashing numerical files or claiming a current byte audit. `read_values`
decodes selected typed Domain/value partitions without implicit byte audits.
Bounded value reads validate persisted canonical Domain schemas and select
calendar/dynamic membership observation rows before constructing a Domain;
they must not validate or sort the complete historical coordinate grid for each
small value window. Keep invalid null observation keys visible to typed validation.
Static asset vectors remain complete. Empty calendar selections return typed
empty values/traces after consumed-schema validation, without inventing a date.
Unbounded reads and explicit integrity APIs still consume the complete Domain;
out-of-window duplicate coordinates remain an explicit full-audit concern.
This view optimization changes no numerical receipt, policy or interval identity.
`evidence(channels=...)` reads only requested auxiliary channels. Explicit full
integrity/update verification covers every saved channel and compares optional
interval indexes to original evidence. Ordinary exact execution uses published
metadata/lazy typed plans; only missing-node consumption loads numerical inputs.
Optional interval proofs include full calendar, membership, empty days, values and
trace/type channels. Completed finite/checkpoint paths with unknown proof or
unprovable relevant parameter/policy scope require explicit maintenance, never
silent historical replay. Stateful contracts declare checkpoint_history read
halos and optional checkpoint_validator structural checks. Reuse original typed
value references and cumulative trace carry; preserve full fit calendars/anchors.
Same-boundary receipt changes with proven identical input perform no source read
or kernel. Historical `index_plan/build_index` maintenance is explicit
and never rewrites manifests. Finite read-context audit proofs expire on exit.
Graph revision guards always read fresh committed state.
Initialization enables WAL explicitly; metadata and graph reads use mode=ro,
query_only transactions and never create graphs or perform recovery.
reference_publication fences Core state while an application commits an immutable
receipt reference in its own transaction; no distributed commit or rollback.
Cleanup preserves committed historical evidence and only reclaims verified
unreferenced completed generations. New storage rejects old schemas.

Run standalone temporary-store tests and owner checks before BT/Workbench.
See [architecture](../../docs/en/architecture.md) for callable examples.

CoreStore.inspect is a public read-only readiness contract for application setup.
Missing storage is uninitialized; mismatched versions, tables or required columns
are incompatible. Neither inspection nor construction initializes or recovers
storage. The public inspection.open_metadata_snapshot primitive reads stable
metadata/WAL copies in a private system temporary directory; source SQLite/WAL/
SHM remain unchanged. BT may reuse this public Core mechanism. Temporary copies
are removed after inspection and never become a second storage authority.
Active/hot rollback journals are unreadable without explicit owner recovery;
never accept spilled uncommitted schema pages. Include journal identity in
snapshot stability checks; invalidated zero-header PERSIST journals are readable.

`CoreStore.inspect(runtime=True)` and `open_metadata_snapshot(..., runtime=True)`
explicitly use ordinary mode=ro SQLite coordination, query_only and one pinned
committed read transaction, respecting WAL and concurrent writer locks. Runtime
callers permit SQLite WAL/SHM coordination; no copy, initialization or recovery.
Abandoned hot journals requiring recovery fail closed on the read-only connection.
Default offline inspection remains source-untouched and rejects active/hot journals.

Explicit `plan_input_proof_rebind` validates frozen original input projections
and registers immutable Core-owned admission metadata; its planning name does
not make it a read-only preview. Normal Graph planning/prerequisite paths never
call it. `apply_input_proof_rebind` preserves original values/receipts and checks
admission before cache reuse and at publication. All reads/audits of rebound
receipts require the complete retained original/admission chain. Source owners
prove exact consumed values, coordinates, type and traces; Core verifies original
finite keys and parameter contexts. Never infer input equivalence from output
equality or fabricate old source bindings.

Use public `Node.source_interval_identity` to bind external source tokens to
actual interval coordinates, types and traces. Bounds are `datetime.date`.
Operation-local proof memos distinguish immutable Domain objects and source
tokens; equal caller-supplied opaque Domain IDs do not prove equal membership.

Domain reference publication selects an original-content-bound indexed receipt
among registered receipts with the same Domain identity, using the same owner
selection in preparation and the closing writer. Validate the actual ordered
coordinates and binding; missing proofs require explicit index maintenance and
corrupt selected proofs fail. Never decode old Domain files to warm an index.

The generic undeclared-history branch rejects a cache miss for completed work
in the same registered context and numerical implementation unless changed
relevant parameters are proven. Missing completion authority or unknown old
parameter bindings require explicit recovery/maintenance before any input load
or kernel. Independent new contexts retain initialization semantics.
`top_n`, `equal_weight`, public Identity and EqualWeight Prediction factories
declare pointwise history=0; their append output/read interval contains only
new sessions. The five supervised Prediction factories declare independent-period
checkpoint work. `checkpoint_periods` selects new/revised and just-matured source
periods; the same Runtime first updates statistics, then calculates exactly their
dependent predictions. Keep the complete period axis, original IC or regression
records, fit-window evidence, source day proofs and arbitrarily delayed pending
labels. A fixed `window` is never a bound on label availability. Missing original
statistics requires explicit evidence maintenance; it cannot be manufactured from
prediction values or reconstructed by silently refitting historical periods.
Statistics caches are distinct from complete numerical receipts and survive
interrupted prediction/publication. Filtered original value references retain
all unchanged output days and immutable files during local repairs.

Rebalance declares checkpoint_history=0 and checkpoint_calendar_role="calendar".
The actual named auxiliary Node Domain supplies the complete anchored date axis,
including empty-universe sessions. Validate it against execution coverage before
numerical work and preserve its completed prefix in checkpoint policy; an opaque
Domain ID or dates inferred from weight rows cannot replace that axis. Proven
continuation reads and calculates only new sessions. Legacy completions without
this checkpoint/calendar authority need explicit maintenance.
`CoreStore.read_checkpoint` loads only state; `compose_artifact` binds a registered
unique dated artifact and same-schema nonoverlapping tail. Publication fences the
original content/channel/file authority and retains old artifact references;
ordinary continuation never decodes or rewrites historical decision tables.

`describe_artifact` and `artifact_interval_identity` inspect committed metadata
only. `read_artifact(start=..., end=...)` consumes typed bounded values with
Parquet predicate pushdown and does not claim a current byte audit. Missing
dated-artifact indexes require explicit `artifact_index_plan`/`build_artifact_index`
maintenance bound to the original content; normal reads never decode history to
backfill proofs. `publish(value_references=...)` also accepts mappings with
original `identity` and exact `start`/`end`, retaining only complete selected days.
Composition preserves ordered complete calendar/Domain coverage and the original
authority/file-existence fences at publication; it does not rewrite unchanged
numerical partitions or implicitly hash historical bytes.
