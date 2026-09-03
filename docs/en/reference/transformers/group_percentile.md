# `group_percentile`

Return average-tie percentile ranks within each declared group and date.

## Signature

```python
group_percentile(source, *, group, name=None, metadata=None)
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
group_percentile(source, group=group)
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 3 | 6 | 10 |
### Panel parameter: group

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | tech | tech | bank | bank |

### Output

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 0.5 | 1 | 0.5 | 1 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

Each date cross-section is calculated independently, so values from other dates cannot influence the result.

The keyword-only `group` Panel participates in graph topology and availability tracing; missing group labels are excluded.
