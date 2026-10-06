# `anscombe`

Translate each date cross-section to a zero minimum, then apply the Anscombe square-root transform.

## Contract

Registry: `bagelquant_core.operator.variance_stabilization.anscombe`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
anscombe(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
anscombe(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | -2 | 0 | 1 |
| 2024-01-03 | -1 | 0.5 | 2 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1.22474 | 3.08221 | 3.67423 |
| 2024-01-03 | 1.22474 | 2.73861 | 3.67423 |
