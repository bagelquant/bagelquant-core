# Code Convention

## General Principles

- Use `Panel` for concrete data.
- Use `Graph` for lazy derived logic.
- Put domain operations in transformer and composer modules.
- Prefer function-style APIs.
- Keep operations deterministic and stateless.
- Do not mutate input frames in place.
- Keep `Panel` immutable at the public boundary.

## Public API Rule

```text
Transformer: Panel | Graph -> Graph
Composer:    (Panel | Graph, ...) -> Graph
Execution:   Graph -> Panel output
```

Do not add one `Graph` method per operation.

## Custom Operations

Use decorators:

```python
@transformer
def rolling_mean(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    return frame.rolling(window).mean()


@composer
def average(*frames: pd.DataFrame) -> pd.DataFrame:
    return sum(frames) / len(frames)
```

Operation configuration must be deterministic and serializable.

## Internal Layers

```text
Panel inputs
    -> operation functions
    -> Graph logic nodes
    -> internal execution runtime
    -> cached Panel outputs
```

## Style

- Use type hints.
- Use structured docstrings for public APIs.
- Follow Black-compatible formatting and an 88-character line length.
- Use `logging`, not `print`, in core logic.
- Raise descriptive exceptions and fail fast.

## Testing

Add focused tests for:

- Panel validation
- Built-in operation behavior
- User-defined decorated operations
- DAG validation
- Alignment
- Cache reuse
- Hash reuse for already-aligned inputs
- Intermediate `Graph.output` population
- Graph specification stability
