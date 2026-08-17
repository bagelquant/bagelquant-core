# `coalesce`

Return the first non-missing value from the supplied inputs for each cell.

## Signature

```python
coalesce(*frames, name=None, metadata=None)
```

## Parameters

**frames** : Panel | Graph
: One or more numeric `Panel` or single-output `Graph` inputs.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
coalesce(input_1, input_2)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### input_1

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | missing |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | missing |
| 2024-01-03 | c | inf |
### input_2

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 10.0 |
| 2024-01-02 | b | 30.0 |
| 2024-01-02 | c | 50.0 |
| 2024-01-03 | a | 20.0 |
| 2024-01-03 | b | 40.0 |
| 2024-01-03 | c | 60.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 30.0 |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | 20.0 |
| 2024-01-03 | b | 40.0 |
| 2024-01-03 | c | inf |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.
