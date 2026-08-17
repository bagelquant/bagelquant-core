# `log`

Return the natural logarithm of positive values and mask invalid inputs.

## Signature

```python
log(source, *, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
log(source)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | -2.0 |
| 2024-01-02 | b | 0.0 |
| 2024-01-02 | c | 1.0 |
| 2024-01-03 | a | -1.0 |
| 2024-01-03 | b | 0.5 |
| 2024-01-03 | c | 2.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | missing |
| 2024-01-02 | b | missing |
| 2024-01-02 | c | 0.0 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | -0.6931471805599453 |
| 2024-01-03 | c | 0.6931471805599453 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
