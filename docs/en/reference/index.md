# Operation reference

BagelQuant operations build deterministic lazy graphs from sparse long-form
`Panel` inputs.

- [Transformer reference](./transformers/index.md): 98 public operations
- [Composer reference](./composers/index.md): 24 public operations

The reference pages are generated from the exported API and curated
documentation metadata. Regenerate them after changing the operation catalog:

```bash
uv run python scripts/generate_operator_reference.py
```
