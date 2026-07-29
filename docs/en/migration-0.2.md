# Migrating from 0.1.x to 0.2.0

Core 0.2 changes the execution model from eager dense Panels at every node to
sparse Polars plans with explicit materialization barriers.

- `Panel.from_domain` accepts `DataFrame | LazyFrame`; it no longer densifies
  construction.
- Use `Panel.collect(dense=False)` for sparse output. `Panel.data` remains the
  dense compatibility API.
- Use `Graph.compile(spec)` and reuse `CompiledGraph` for repeated input
  batches.
- Custom decorators without a contract run as safe dense eager barriers.
- Custom operations consuming trace columns must declare `trace_rule`.
- Cache identities are caller assertions. Reuse an explicit identity only for
  immutable equivalent input content; otherwise omit it and use the safe
  instance token.
- Persisted cache and invalidation behavior are unchanged and remain owned by
  the calling data layer.
