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
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### input_1

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 3 |
| 2024-01-03 | missing | missing | inf |
### input_2

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 10 | 30 | 50 |
| 2024-01-03 | 20 | 40 | 60 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 30 | 3 |
| 2024-01-03 | 20 | 40 | inf |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.
