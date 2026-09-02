# `broadcast_by_time`

Broadcast one source value per date to every keyed row present in a second Panel.

## Signature

```python
broadcast_by_time(source, like, *, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**like** : Panel | Graph
: Stock-domain `Panel` or single-output `Graph` whose dated keys define the broadcast output.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
broadcast_by_time(input_1, input_2)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### input_1

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | missing |
| 2024-01-02 | b | missing |
| 2024-01-02 | c | missing |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | missing |
| 2024-01-03 | c | missing |
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
| 2024-01-02 | a | missing |
| 2024-01-02 | b | missing |
| 2024-01-02 | c | missing |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | missing |
| 2024-01-03 | c | missing |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.
