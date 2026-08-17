# `ewm_std`

Return the per-asset exponentially weighted moving standard deviation in time order.

## Signature

```python
ewm_std(source, *, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, bias=False, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**com** : float | None, default `None`
: Center-of-mass decay parameter. Supply exactly one decay parameter.
**span** : float | None, default `None`
: Span decay parameter. Supply exactly one decay parameter.
**halflife** : float | None, default `None`
: Half-life decay parameter. Supply exactly one decay parameter.
**alpha** : float | None, default `None`
: Smoothing or regularization parameter, depending on the operation.
**min_periods** : int, default `0`
: Minimum number of observations required to produce a value.
**adjust** : bool, default `True`
: Whether to divide by the decaying adjustment factor.
**ignore_na** : bool, default `False`
: Whether missing values are ignored when calculating weights.
**bias** : bool, default `False`
: Whether to use the biased exponentially weighted variance calculation.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
ewm_std(source, span=2.0)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 2.0 |
| 2024-01-03 | a | 2.0 |
| 2024-01-03 | b | 3.0 |
| 2024-01-04 | a | 4.0 |
| 2024-01-04 | b | 5.0 |
| 2024-01-05 | a | 7.0 |
| 2024-01-05 | b | 8.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 0.0 |
| 2024-01-02 | b | 0.0 |
| 2024-01-03 | a | 0.7071067811865475 |
| 2024-01-03 | b | 0.7071067811865475 |
| 2024-01-04 | a | 1.5689290811054721 |
| 2024-01-04 | b | 1.5689290811054721 |
| 2024-01-05 | a | 2.6253204932555745 |
| 2024-01-05 | b | 2.6253204932555745 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
