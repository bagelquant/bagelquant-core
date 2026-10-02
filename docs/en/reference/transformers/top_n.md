# `top_n`

Select exactly N finite scores, breaking ties by asset ID; incomplete cross-sections remain unavailable.

## Signature

```python
top_n(source, *, count=20, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**count** : int, default `20`
: Exact number of selected assets, with ties ordered by asset_id.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
top_n(source, count=2)
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0 | 1 | 1 |
| 2024-01-03 | 0 | 1 | 1 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
