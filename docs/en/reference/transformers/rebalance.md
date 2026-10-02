# `rebalance`

Select complete target snapshots on an explicit calendar anchored at Data Start.

## Signature

```python
rebalance(source, *, every=5, anchor='data_start', calendar, data_start, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**every** : int, default `5`
: Positive number of trading sessions between decisions.
**anchor** : str, default `'data_start'`
: Decision anchor: data_start.
**calendar** : Panel | Graph
: Explicit auxiliary trading calendar supplied by the application.
**data_start** : str
: Global Data Start; the first following trading session anchors decisions.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
rebalance(source, calendar=calendar, every=1, data_start='2024-01-02')
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |
### Panel parameter: calendar

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 2 | 2 |
| 2024-01-03 | 2 | 1 | 4 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
