# `canonicalize_values`

Round numerical values to explicit significant digits at a semantic boundary.

## Signature

```python
canonicalize_values(source, *, significant_digits=14, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**significant_digits** : int, default `14`
: Positive number of significant digits at an explicit numerical boundary.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
canonicalize_values(source)
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
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
