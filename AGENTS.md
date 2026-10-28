# BagelQuant Core agent entrance

Read [CONTRIBUTING.md](CONTRIBUTING.md), [README.md](README.md) and
[pyproject.toml](pyproject.toml), then the affected technical contracts routed
by the contributor guide. [Owner boundaries](docs/en/engineering/development.md) remain authoritative.

Core owns typed Nodes, Operators, Graphs, numerical/ML primitives and CoreStore artifacts. It has no BagelQuant package dependencies. Callers supply neutral inputs, paths, immutable evidence and resource budgets.

Inspect tracked and untracked Git changes before editing. Preserve unrelated
work, Git metadata and historical evidence. Use Python 3.13 and uv; tests use
isolated temporary roots and synthetic inputs or fake providers.
Run the applicable owner checks in CONTRIBUTING.md and report actual results.
Publication, commits, provider calls and real-data/service changes require
explicit authorization. Keep generated references owned by their generators.
