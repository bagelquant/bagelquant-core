# BagelQuant Core

Standalone typed computation for quantitative research: Domain, Node, Operator,
Graph, safe DSL, incremental execution and caller-located persistent caches.
Core has no BagelQuant package dependency and no global scheduling policy.

Python 3.13; install with `uv add bagelquant-core`. Optional extras: `ml`, `optimizer`.

- [Architecture and runnable standalone example](docs/en/architecture.md)
- [中文架构](docs/cn/architecture.md)
- [Generated Operator catalog](docs/en/reference/operators/index.md)
- [AI workflow](.ai/README.md)

Development: `uv sync --all-extras`, `uv run ruff check .`,
`uv run python -m pytest`, and
`uv run python scripts/generate_operator_reference.py --check`.
On macOS, install the CPU ML runtime's OpenMP dependency with
`brew install libomp` before validating all extras.

Version 0.11 uses new API and storage schemas. It provides no compatibility reader
or migration. Real data and service cutover are separate operations.

Supported platforms: macOS and Linux. Windows support is retired.
