# `sum_frames`

Return the element-wise sum across all aligned input panels.

## Signature

```python
sum_frames(*frames, name=None, metadata=None)
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
sum_frames(input_1, input_2)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### input_1

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 4.0 |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | 2.0 |
| 2024-01-03 | c | 8.0 |
### input_2

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 2.0 |
| 2024-01-02 | c | 2.0 |
| 2024-01-03 | a | 2.0 |
| 2024-01-03 | b | 1.0 |
| 2024-01-03 | c | 4.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 2.0 |
| 2024-01-02 | b | 6.0 |
| 2024-01-02 | c | 5.0 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | 3.0 |
| 2024-01-03 | c | 12.0 |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.
