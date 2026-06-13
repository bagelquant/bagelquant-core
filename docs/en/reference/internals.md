# Internal Documentation

This page describes internals that are useful for maintainers. Users should
prefer the public API docs unless they are extending the package.

## Runtime Model

Every operation call creates an internal node with:

- Parent dependencies
- Operation identity
- Serializable configuration
- User-facing name and metadata
- Cached output after execution

The public `Graph` object is a wrapper around this lazy node chain.

## Execution

`Graph.compute()` creates an execution runtime and evaluates dependencies
recursively. Within one compute call, shared upstream nodes are executed once.
The runtime validates domain compatibility before multi-input composition,
applies operation functions to Polars frames, then wraps results back into
`Panel` objects.

Dynamic universe masks are reapplied after derived computations so inactive
cells cannot leak into later operations.

## Hashing And Caching

Panels and operation specs participate in deterministic cache keys. The current
cache is in-memory for one execution run. Persistent caches, incremental
invalidation, and parallel scheduling are future extensions.

## Operation Registry

Built-in operation modules define transformer and composer functions using the
same decorators available to users. Generated reference docs are produced from
the exported operation catalog:

```bash
uv run python scripts/generate_operator_reference.py
```

Regenerate the reference after adding, removing, or changing operation
metadata.

## Module Boundaries

- `panel` owns domain alignment and immutable data containers.
- `graph` owns lazy graph handles and user-facing compute/output access.
- `execution` owns dependency evaluation.
- `transformer` owns unary frame operations.
- `composer` owns multi-input frame operations.
- `registry` and `_operation` own operation metadata and decorators.

