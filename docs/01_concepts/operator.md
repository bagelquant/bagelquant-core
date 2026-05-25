# Operator

## Overview

An Operator is a unary transformation on a Panel.

```text
Panel → Operator → Panel
```

Operators are the fundamental computational building blocks in BagelQuant.

They transform a single input Panel into a single output Panel while preserving the Panel abstraction.

Every Operator consumes exactly one Panel and produces exactly one Panel.

---

## Definition

Mathematically:

```text
Operator : P → P
```

where:

```text
P = Panel
```

An Operator can be viewed as a function:

```python
output_panel = operator(input_panel)
```

---

## Philosophy

Operators should be:

- Small
- Reusable
- Deterministic
- Composable

Rather than creating large monolithic transformations, complex workflows should be constructed by chaining simple Operators together.

Preferred:

```text
Factor
  ↓
ZScore
  ↓
Neutralize
  ↓
Winsorize
```

Not preferred:

```text
FactorNormalizationPipeline
```

Small Operators improve:

- Transparency
- Reusability
- Testing
- Debugging
- Graph visualization

---

## Characteristics

Every Operator must satisfy the following properties.

### Single Input

An Operator accepts exactly one input Panel.

```text
Input Panel
      ↓
   Operator
      ↓
Output Panel
```

---

### Single Output

An Operator produces exactly one output Panel.

```text
Panel → Panel
```

---

### Stateless

Operators should not maintain internal state between executions.

Given the same input:

```python
operator(panel)
```

the output should always be identical.

---

### Deterministic

For a given input Panel and configuration:

```python
rank(panel)
```

the output must always be the same.

This property is critical for:

- Reproducibility
- Caching
- Incremental computation

---

### Immutable

Operators never modify input Panels.

Instead:

```python
output_panel = operator(input_panel)
```

Input Panels remain unchanged.

---

## Examples

### Example 1: Cross-Sectional Rank

```text
Price Panel
      ↓
     Rank
      ↓
Ranked Price Panel
```

Input:

```text
AAPL    100
MSFT     50
NVDA    200
```

Output:

```text
AAPL    0.50
MSFT    0.00
NVDA    1.00
```

---

### Example 2: Z-Score

```text
Factor Panel
      ↓
    ZScore
      ↓
Normalized Factor
```

Used to normalize factor distributions.

---

### Example 3: Rolling Mean

```text
Price Panel
      ↓
 RollingMean(20)
      ↓
Smoothed Price Panel
```

Computes moving averages across time.

---

### Example 4: Neutralization

```text
Factor Panel
      ↓
 MarketCapNeutralize
      ↓
Neutralized Factor
```

Removes exposure to market capitalization.

---

### Example 5: Portfolio Normalization

```text
Raw Weights
      ↓
 Normalize
      ↓
Portfolio Weights
```

Ensures portfolio weights satisfy constraints.

---

## Operator Categories

Operators can be grouped into several categories.

---

### Time-Series Operators

Operate along the time dimension.

Examples:

```text
RollingMean
RollingStd
Returns
Delta
Lag
EMA
```

Example:

```text
Price
  ↓
Returns
```

---

### Cross-Sectional Operators

Operate across assets at a given timestamp.

Examples:

```text
Rank
ZScore
Winsorize
Neutralize
```

Example:

```text
Factor
  ↓
Rank
```

---

### Statistical Operators

Perform statistical transformations.

Examples:

```text
Standardize
Clip
Scale
Transform
```

---

### Portfolio Operators

Transform predictions into investable signals.

Examples:

```text
Normalize
CapWeight
ExposureAdjustment
LeverageControl
```

Example:

```text
Prediction
    ↓
Normalize
    ↓
Weights
```

---

## Operator Chains

Operators are designed to be chained together.

Example:

```text
BM Ratio
    ↓
 ZScore
    ↓
Neutralize
    ↓
Winsorize
    ↓
 BM Factor
```

Each step remains visible and independently testable.

---

## Operator vs Composer

A common design question is:

> Should this be an Operator or a Composer?

Rule:

### Operator

Uses exactly one Panel.

```text
P → P
```

Examples:

```text
Rank
ZScore
RollingMean
Normalize
```

---

### Composer

Uses multiple Panels.

```text
(P₁, P₂, ..., Pₙ) → P
```

Examples:

```text
Add
Divide
OLS
NeuralNetwork
Optimizer
```

---

## Dependency Behavior

Operators create edges within the graph.

Example:

```text
Price Panel
      ↓
 Returns
      ↓
 Momentum
```

Dependency chain:

```text
Price → Returns → Momentum
```

If Price changes:

```text
Price Updated
```

only downstream nodes are recomputed.

---

## Caching

Operators are pure transformations.

Therefore:

```python
rank(price_panel)
```

can be safely cached.

Benefits:

- Faster execution
- Incremental updates
- Reduced computation cost

---

## Design Guidelines

When creating a new Operator:

### Keep It Small

Preferred:

```text
Rank
```

Not:

```text
RankAndNeutralizeAndScale
```

---

### Keep It Deterministic

Avoid hidden randomness.

---

### Keep It Reusable

Operators should solve a single transformation problem.

---

### Preserve Panel Semantics

Input:

```text
Panel
```

Output:

```text
Panel
```

Never break the Panel abstraction.

---

## Examples in Quantitative Research

### Factor Construction

```text
BM Ratio
    ↓
 ZScore
    ↓
Neutralize
    ↓
 BM Factor
```

---

### Signal Processing

```text
Prediction
      ↓
 Clip
      ↓
 Normalize
      ↓
 Signal
```

---

### Portfolio Construction

```text
Prediction
      ↓
 CapWeight
      ↓
 Portfolio Weights
```

---

## Summary

Operators are unary Panel transformations.

```text
Panel → Panel
```

They represent the smallest reusable computational unit in BagelQuant.

Operators are deterministic, stateless, immutable, and composable.

By chaining simple Operators together, complex quantitative research workflows can be expressed as transparent and reusable DAGs.

