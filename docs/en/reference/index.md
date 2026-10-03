# Operation reference

BagelQuant operations build deterministic lazy graphs from sparse long-form
`Panel` inputs. All operations use the single `Operator` registry and
`OperationNode` execution contract. Transformer/composer headings describe
input arity; they do not create separate execution systems.

- [Transformer reference](./transformers/index.md): 105 public operations
- [Composer reference](./composers/index.md): 27 public operations

The reference pages are generated from the exported API and curated
documentation metadata. Regenerate them after changing the operation catalog:

```bash
uv run python scripts/generate_operator_reference.py
```
