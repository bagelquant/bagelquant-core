# `group_mean`

Replace each value with its declared group's same-date arithmetic mean.

## Signature

```python
group_mean(source, *, group, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**group** : Panel | Graph
: Matching `CategoryPanel` containing row-wise group labels.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
group_mean(source, group=group)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 3.0 |
| 2024-01-02 | c | 6.0 |
| 2024-01-02 | d | 10.0 |
### Panel parameter: group

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | tech |
| 2024-01-02 | b | tech |
| 2024-01-02 | c | bank |
| 2024-01-02 | d | bank |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 2.0 |
| 2024-01-02 | b | 2.0 |
| 2024-01-02 | c | 8.0 |
| 2024-01-02 | d | 8.0 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

Each date cross-section is calculated independently, so values from other dates cannot influence the result.

The keyword-only `group` Panel participates in graph topology and availability tracing; missing group labels are excluded.
