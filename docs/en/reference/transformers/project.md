# `project`

Keep source values only where the key-aligned binary Panel equals one.

## Signature

```python
project(source, *, binary, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**binary** : Panel | Graph
: Binary projection input. Cells equal to `1` are retained; other cells are masked.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
project(source, binary=binary)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 4.0 |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | 2.0 |
| 2024-01-03 | c | 8.0 |
### Panel parameter: binary

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 1.0 |
| 2024-01-02 | c | 1.0 |
| 2024-01-03 | a | 0.0 |
| 2024-01-03 | b | 0.0 |
| 2024-01-03 | c | 0.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 4.0 |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | missing |
| 2024-01-03 | c | missing |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
