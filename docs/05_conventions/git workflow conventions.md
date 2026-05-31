# Git Workflow Convention

## Philosophy

Use a lightweight trunk-based workflow:

- Keep `main` stable and runnable.
- Use short-lived, single-purpose branches.
- Make each pull request represent one coherent concept.
- Run tests before merge.

## Branch Names

Use lowercase words separated by hyphens:

```text
feature/panel-boundary
refactor/function-operations
fix/graph-output-cache
docs/execution-model
```

## Commits

Use Conventional Commit summaries:

```text
feat: add panel output caching
refactor: use function-style graph operations
fix: handle constant zscore inputs
docs: update graph execution model
```

Keep summaries imperative and concise.

## Pull Requests

Include:

```text
## What
## Why
## How
## Testing
## Notes
```

Use squash merge after review and passing tests.
