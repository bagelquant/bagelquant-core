# Contributing to BagelQuant Core

Core owns typed Nodes, Operators, Graphs, numerical/ML primitives and CoreStore artifacts. It has no BagelQuant package dependencies. Callers supply neutral inputs, paths, immutable evidence and resource budgets.

## Setup and supported platform

Use macOS, Python 3.13 and uv. Read [README.md](README.md),
[pyproject.toml](pyproject.toml) and the affected technical contracts below.
Install the declared development dependencies with `uv sync --locked --all-extras`.
Run commands from this repository. Validation runs locally; GitHub Actions is disabled.
This contributor guide and the package's tests are available in a standalone checkout.

## Technical contracts by topic

The engineering documents specify owner invariants; the linked manuals explain
usage. Keep one authority for each formula, schema, model and execution path.
Verify a disagreement against current source and representative tests before
updating the affected documentation. Historical evidence keeps its original meaning.

| Task topic | Owner rules | Relevant docs |
| --- | --- | --- |
| Node/Domain, neutral input conversion, membership, Operator, typed Prediction, catalog | [Nodes and operators](docs/en/engineering/nodes-operators.md) | [Concepts](docs/en/concepts.md), [Operator](docs/en/reference/concepts/operator.md) |
| Logical graph, numerical artifact/cache/checkpoint persistence, canonical identity, materialization, causal history, execution/resources | [Graph and runtime](docs/en/engineering/graph-runtime.md) | [Architecture](docs/en/architecture.md), [Execution](docs/en/reference/concepts/execution.md), [Performance](docs/en/performance.md) |
| Generic ML, mature labels, deterministic training, IC weighting, OLS, optimizers | [Numerical and training contracts](docs/en/engineering/numerical-training.md) | [Public API](docs/en/reference/public-api.md), [Internals](docs/en/reference/internals.md) |
| Public operation/export/doc metadata change | [Nodes and operators](docs/en/engineering/nodes-operators.md), [development](docs/en/engineering/development.md) | [Operation reference](docs/en/reference/index.md) |

## Engineering and validation

Inspect tracked and untracked Git changes and preserve unrelated work and Git
metadata. Keep changes focused, dependencies explicit, typed boundaries validated,
ordering deterministic, and errors actionable. Use logging in library/application
logic and vectorized Polars/NumPy where appropriate. Remove obsolete paths and
their exports/tests/docs together; add compatibility only for a concrete requirement.

Use isolated temporary roots and synthetic inputs or fake providers for tests.
Keep test fixtures, caches and output outside the checkout. Preserve real data,
credentials, authored research, historical receipts and recovery evidence.
Performance work preserves numerical/PIT contracts. Formula changes need a
hand-checkable example and a simple reference; cover sparse/missing/duplicate/empty
inputs. Calculation changes must prove bounded new/affected work and historical
reuse under the owner contracts. Distinguish observation, information cutoff,
signal-effective and execution dates; keep stable typed keys and missingness.

For code changes, run the applicable checks:

```bash
uv run ruff check .
uv run python -m pytest
uv run python scripts/generate_operator_reference.py --check
```

Choose focused checks for prose changes. Test the owner before affected consumers
when a public boundary changes, and report actual results and unrun checks.
Regenerate generated references, catalogs, OpenAPI and API types from their owner;
never edit them manually. Update the owner documentation with public behavior,
formula, schema and workflow changes. Active authored documentation is English;
preserve original sources, market identifiers, Unicode data and historical evidence.

Keep main stable and branches focused. Use imperative Conventional Commit
summaries of at most 72 characters. Keep credentials, datasets, databases,
environments and build output out of Git. Commits, pushes, PRs, merges, releases,
deployment, services, provider calls and real-data/governance changes require
their own explicit authorization; validation does not authorize those operations.
