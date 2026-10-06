# Core development rules

## Preparation and boundaries

- Inspect `git status --short --branch` including untracked changes before editing;
  read README, manifest and affected docs. Work in the owning repository and preserve
  unrelated user edits. Never remove Git metadata, merge histories or absorb repos.
- Supported platforms are macOS and Linux; Windows support is retired.
- Use Python 3.13 and `uv`. Choose cross-platform paths/APIs; account for separators,
  casing, line endings, permissions, shell syntax and environment conventions on
  macOS/Linux. Use `pathlib`; never persist developer-specific absolute paths.
- Core has no Data/BT/Workbench dependency. Put reusable mechanics here, consumer
  policy downstream, and declare any necessary dependency in the owning manifest.

## Package target and staged refactor

- Target ownership: Domain/Node conversion, numerical transforms/ML, graph
  execution and generic numerical artifact/materialization/cache/checkpoint
  persistence belong to Core. CoreStore implements caller-located SQLite/Parquet persistence; applications use public receipts.
- Accept neutral frames/schema/availability/immutable identity evidence through
  public APIs. Data owns dataset storage and frozen input evidence; Core does
  not query Data or provider state. BT owns account/evaluation artifact storage.
- Workbench owns China semantics, app metadata, research governance and task
  orchestration, referencing backend artifacts. Generic conversion, computation,
  numerical persistence and reuse proofs must not be reimplemented there.
- Follow rules -> Data -> Core -> BT -> Workbench -> new database/service restart.
  Stages 2 and 3 implement Data and Core; BT and full Workbench cleanup remain later stages. Breaking refactors remove old paths without compatibility.

## Implementation

- Prefer the smallest complete system and existing lower-level primitives. Keep
  one authoritative API/model/state/execution path per concern. Delete dead code
  rather than add adapters, aliases, flags or parallel implementations.
- Do not keep deprecated readers/routes/DSL aliases or migration shims unless
  an explicit compatibility window is requested. Add abstractions, dependencies,
  configuration or persisted state only for a current concrete responsibility.
- Keep dependencies and boundaries visible. Remove a feature's exports, tests,
  docs, configuration and generated references together; preserve real shared
  data, recovery evidence, user work, credentials, environments and caches.
- Use deterministic composable logic, descriptive names and typed public
  boundaries; follow surrounding structured docstrings. Validate boundaries
  with actionable errors and never silently swallow exceptions.
- Use `logging`, not `print`, in library logic. Prefer vectorized Polars/NumPy;
  row-wise loops need justification and measurement. Ordering/grouping are explicit.
- Performance changes preserve contracts unless a behavior change is requested
  and documented. Formula changes need a hand-checkable example and regression
  test; compare optimized output with a simple reference where practical.
- Cover sparse calendars/membership, missing values, duplicate keys, empty
  frames and non-trading dates where relevant. Tests use temporary roots and
  synthetic/mock inputs and never touch the real workspace data root/provider quota.

## Verification and delivery

- Run `uv run ruff check .` and `uv run python -m pytest` for code changes;
  choose focused checks for documentation-only work. Never edit generated docs
  directly: run `uv run python scripts/generate_operator_reference.py` after
  public operation/export/metadata changes and include generated results.
- State cross-repo public contract changes. Keep edits independently coherent,
  update bounds/versions only when required, test Core first then affected BT
  and Workbench consumers, and regenerate Workbench DSL catalogs separately.
- For a package refactor, name one owner per capability/artifact and prove the
  public API works without Workbench using synthetic inputs and temporary roots.
  Check dependency direction, remove superseded paths when authority moves,
  and verify numerical artifact persistence/reuse separately from app metadata.
  A missing backend API is work for its owner, not a generic Workbench workaround.
- Major architecture changes update applicable AGENTS and owner rules in the
  same change; update root instructions if cross-repository boundaries change.
- Do not commit caches, credentials, databases, provider data, environments,
  build outputs, research artifacts or local task records. Use stable `main`
  and short-lived single-purpose branches; honor session branch instructions.
  Conventional Commit summaries are imperative and at most 72 characters.
- Commit/push/PR/merge/release/publication/deployment/service installation,
  real-data updates and governance transitions each require explicit request.
  Commit components separately before explicitly authorized gitlink updates.
  Package publication also verifies version, build contents, registry,
  credentials and version absence; report exact uploaded version/artifacts.
- Report changed behavior/why, repository-specific checks/results, unrun checks
  with reasons, and contract/version/data/operational caveats. Distinguish existing
  failures from introduced failures; never claim a check passed unless run.
