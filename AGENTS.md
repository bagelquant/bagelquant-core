# Core agent entry

This repository owns Domain, typed Node, Operator and Graph, safe DSL, deterministic
numerical/ML primitives and CoreStore graph/cache/checkpoint/artifact persistence.
Caller-supplied paths, source evidence, cutoffs and resource budgets are explicit.
Core has no BagelQuant package dependency; Workbench owns global scheduling and
machine policy. Do not add providers, accounts, China rules or governance here.

Before work, read [`.ai/README.md`](.ai/README.md), the mandatory development
rules, the affected topic rules, [`README.md`](README.md), [`pyproject.toml`](pyproject.toml)
and relevant local docs. Inspect tracked and untracked Git changes first.
Keep this entry short; detailed owner contracts live under `.ai/rules/`.

- Use Python 3.13 and `uv`; run commands from this repository.
- Preserve unrelated work, Git metadata, credentials, environments and data.
- Tests use isolated temporary roots and synthetic inputs; never read or
  mutate the workspace's real data root or call a real provider for validation.
- Do not commit, push, create PRs, merge, release, deploy, install services,
  update real data or change governance unless explicitly requested.
- Keep immutable sparse Nodes and explicit value_type; BT has a typed
  prediction boundary and a separate saved-target boundary.
- Accept neutral input data and immutable evidence through public APIs; Core
  owns conversion and numerical persistence, Data owns frozen input authority.
- Regenerate operation references from their generator after catalog changes;
  coordinate separate Workbench catalog updates when the public contract changes.

For formal work in an integrated workspace, create/resume a root `.ai/tasks/`
record using the workspace CLI. Discover the workspace with
`git rev-parse --show-superproject-working-tree`. If empty, inspect checkout
ancestors as candidates. Accept only a candidate that is its own Git root,
declares the four component paths in `.gitmodules` and has their index gitlinks
(mode `160000`); this checkout must match its exact declared relative owner path.
For every candidate, also verify
root `AGENTS.md`, `.ai/README.md`, `.ai/workflow.md`, `.ai/rules/workspace.md`
and `scripts/ai.py` exist. Read the root workflow and workspace rules; load
cross-repository contracts only for affected topics. Never guess parent paths.
In a standalone checkout, follow these local rules and record plan, validation
and handoff in the conversation; do not create a component task directory.
Plan/read-only mode never writes task records or implementation files.

Validate code changes with `uv run ruff check .` and `uv run python -m pytest`.
Report affected contracts, checks/results, unrun checks and next steps honestly.
