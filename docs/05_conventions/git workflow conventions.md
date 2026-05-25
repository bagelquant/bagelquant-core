# Git Workflow Convention

## 1. Philosophy

This repository uses a **lightweight trunk-based development workflow**:

- `main` is always stable
- All changes go through Pull Requests
- Branches are short-lived and single-purpose
- PRs represent one atomic concept

---

# 2. Branch Structure

## 2.1 Main Branch

### `main`

- Always stable and runnable
- Must pass all tests before merge
- No direct commits allowed
- Only updated via Pull Requests

---

## 2.2 Feature Branches

### Naming:

```plaintext
feature/<short-description>
```

### Examples:

- `feature/panel-core`
- `feature/operator-system`
- `feature/dag-composer`
- `feature/execution-engine`

### Rules:

- One branch = one concept
- Should be merged within 1–7 days
- Avoid mixing unrelated changes

---

## 2.3 Bug Fix Branches

### Naming:

```plaintext
fix/<short-description>
```

### Examples:

- `fix/null-panel-handling`
- `fix/dag-cycle-check`
- `fix/ic-calculation-bug`

---

## 2.4 Refactor Branches

### Naming:

```plaintext
refactor/<short-description>
```

### Examples:

- `refactor/expression-tree`
- `refactor/operator-api`

### Rules:

- No behavior change unless explicitly stated
- Should include structural cleanup or abstraction improvement

---

## 2.5 Experiment Branches (Optional)

### Naming:

```plaintext
experiment/<short-description>
```

### Examples:

- `experiment/new-normalization`
- `experiment/alpha-composer-v2`

### Rules:

- Not guaranteed to merge
- Used for research / prototyping
- Must not break `main` stability assumptions

---

## 2.6 Documentation Branches (Optional)

### Naming:

```plaintext
docs/<short-description>
```

### Examples:

- `docs/dag-design`
- `docs/operator-guide`
- `docs/composer-tutorial`

# 3. Commit Convention

We follow **Conventional Commits**:

## Format:

```plaintext
<type>: <short summary>
```

## Types:

- `feat`: new feature
- `fix`: bug fix
- `refactor`: code restructuring
- `perf`: performance improvement
- `test`: testing
- `docs`: documentation
- `chore`: maintenance

## Examples:

```plaintext
feat: add DAG-based composer engine
fix: handle missing panel input
refactor: simplify operator interface
docs: update DAG design explanation
```

---

## Rules:

- Imperative tone ("add", not "added")
- Keep summary ≤ 72 characters
- Atomic commits preferred

---

# 4. Pull Request Convention

## 4.1 PR Naming

```plaintext
[TYPE] Short description
```

### Examples:

- `[FEAT] Add DAG-based expression system`
- `[FIX] Resolve panel alignment bug`
- `[REFACTOR] Simplify operator abstraction`

---

## 4.2 PR Template

Every PR must include:

```plaintext
## What

What does this PR change?

## Why

Why is this change needed?

## How

High-level implementation details.

## Changes

- …
- …
- …

## Testing

- [ ] unit tests pass
- [ ] example scripts run
- [ ] DAG integrity verified

## Notes

Any design decisions or trade-offs.
```

---

## 4.3 PR Size Rule

A PR must satisfy:

> One PR = One Concept

### Good:

- DAG node system only
- Operator abstraction only

### Bad:

- DAG + execution + IC evaluation in one PR

---

## 4.4 Review Requirement

Before merging:

- Self-review required
- CI must pass (if enabled)

---

## 4.5 Merge Strategy

Use:
> **Squash and Merge**

### Reason:
- Keeps `main` clean
- One PR = one commit
- Easier rollback

---

# 5. Branch Lifecycle

## Workflow:

main
↓
create feature/fix/refactor branch
↓
development
↓
open PR
↓
review + CI
↓
squash merge into main
↓
delete branch

---

# 6. Naming Consistency Rules

## 6.1 Use lowercase + hyphens

Good:
- `feature/dag-composer`
- `fix/null-handling`
Bad:
- `Feature/DAGComposer`
- `fix_null_handling`

---

## 6.2 Be concept-based, not file-based

Good:
- `feature/operator-system`
Bad:
- `feature/operator.py-update`

---

# 7. Release Strategy (Optional but recommended)

If versioning is needed:

## Tag format:

v0.1.0
v0.2.0

## Rule:

- Only `main` can be tagged
- Tag = stable snapshot

---

# 8. Enforcement Rules (Important)

The following are **strong rules**:

1. No direct push to `main`
2. No PR with multiple unrelated concepts
3. No commit without meaningful message
4. No merge without passing CI/tests (if available)
5. Delete branch after merge

# Summary

This repo follows:
> **Trunk-based development + strict PR discipline + concept-level branching**
