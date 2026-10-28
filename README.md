# BagelQuant Core

Standalone typed computation for quantitative research: Domain, Node, Operator,
Graph, safe DSL, incremental execution and caller-located persistent caches.
Core has no BagelQuant package dependency and no global scheduling policy.

Python 3.13; install with `uv add bagelquant-core`. Optional extras: `ml`, `optimizer`.

- [Architecture and runnable standalone example](docs/en/architecture.md)
- [Causal version inputs and bounded replay](docs/en/causal-inputs.md)
- [Generated Operator catalog](docs/en/reference/operators/index.md)
- [Contributor guide and engineering contracts](CONTRIBUTING.md)

Development: `uv sync --all-extras`, `uv run ruff check .`,
`uv run python -m pytest`, and
`uv run python scripts/generate_operator_reference.py --check`.
On macOS, install the CPU ML runtime's OpenMP dependency with
`brew install libomp` before validating all extras.

Version 0.11 uses new API and storage schemas. It provides no compatibility reader
or migration. Real data and service cutover are separate operations.
Core metadata schema 4 adds content-bound transient region certificates and
logical update coverage to indexed durable completions and candidate headers.
Older schemas require an explicitly initialized fresh location; no migration
or compatibility reader is provided. Pointwise regions preserve canonical keys
and arithmetic while persisting their requested and shared boundaries.

Supported platform: macOS. Validation runs locally; GitHub Actions is disabled.

[Calculation records](docs/en/calculation_records.md)
