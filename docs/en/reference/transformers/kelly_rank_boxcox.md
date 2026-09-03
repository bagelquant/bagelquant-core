# `kelly_rank_boxcox`

Percentile-rank each date cross-section, apply Box-Cox, then calculate the rolling Kelly mean-to-variance ratio.

## Signature

```python
kelly_rank_boxcox(source, *, window, lambda_=0, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**window** : int
: Positive trailing-window length in rows.
**lambda_** : float, default `0`
: Box-Cox lambda parameter. Use `0` for the logarithmic limit.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
kelly_rank_boxcox(source, window=2)
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 2 | 3 | 1 |
| 2024-01-03 | missing | 2 | 4 |
| 2024-01-04 | 5 | 1 | 3 |
| 2024-01-05 | 2 | 6 | 1 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | missing | missing |
| 2024-01-03 | missing | -1.4427 | -0.910239 |
| 2024-01-04 | missing | -10.8987 | -2.4663 |
| 2024-01-05 | -2.4663 | -0.910239 | -3.13054 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
