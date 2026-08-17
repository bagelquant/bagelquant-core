# `greater_equal`

Return one where the first input is greater than or equal to the second and zero elsewhere.

## Signature

```python
greater_equal(lhs, rhs, *, name=None, metadata=None)
```

## Parameters

**lhs** : Panel | Graph
: Left-hand numeric `Panel` or single-output `Graph`.
**rhs** : Panel | Graph
: Right-hand numeric `Panel` or single-output `Graph`.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
greater_equal(input_1, input_2)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### input_1

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 0.0 |
| 2024-01-02 | b | missing |
| 2024-01-02 | c | 0.0 |
| 2024-01-03 | a | 1.0 |
| 2024-01-03 | b | 1.0 |
| 2024-01-03 | c | 2.0 |
### input_2

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 0.0 |
| 2024-01-02 | b | 1.0 |
| 2024-01-02 | c | 2.0 |
| 2024-01-03 | a | 0.0 |
| 2024-01-03 | b | 1.0 |
| 2024-01-03 | c | missing |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | missing |
| 2024-01-02 | c | 0.0 |
| 2024-01-03 | a | 1.0 |
| 2024-01-03 | b | 1.0 |
| 2024-01-03 | c | missing |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.

Logical and comparison results are numeric panels containing `1.0` and `0.0`.
