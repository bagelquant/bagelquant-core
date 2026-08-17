# `weighted_sum`

Return the element-wise weighted sum of aligned value and weight panels.

## Signature

```python
weighted_sum(*frames, weights, name=None, metadata=None)
```

## Parameters

**frames** : Panel | Graph
: One or more numeric `Panel` or single-output `Graph` inputs.
**weights** : Sequence[float]
: One numeric weight for each input frame.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
weighted_sum(input_1, input_2, weights=[0.25, 0.75])
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
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 2.5 |
| 2024-01-02 | c | 2.25 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | 1.25 |
| 2024-01-03 | c | 5.0 |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.
