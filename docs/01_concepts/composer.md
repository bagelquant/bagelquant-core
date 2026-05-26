# Composer

## Overview

A Composer combines one or more input Panels into a single output Panel.

```text
(P₁, P₂, ..., Pₙ) → Composer → P
```

Composers represent relationships, interactions, and combinations between Panels.

While Transformers perform unary transformations on a single Panel, Composers are responsible for merging information from multiple Panels.

Composers are used throughout the quantitative investment process:

- Feature engineering
- Factor construction
- Alpha combination
- Prediction modeling
- Portfolio optimization

---

## Definition

Mathematically:

```text
Composer : (P₁, P₂, ..., Pₙ) → P
```

where:

```text
P = Panel
```

A Composer can be viewed as a function:

```python
output_panel = composer(
    panel_1,
    panel_2,
    ...
    panel_n
)
```

---

## Philosophy

Composers are responsible for combining information.

Whenever a computation requires multiple input Panels, it should be represented as a Composer.

Examples:

```text
Book Value
Price
    ↓
 Divide
    ↓
BM Ratio
```

```text
Factor A
Factor B
Factor C
    ↓
Linear Model
    ↓
Prediction
```

```text
Prediction
Risk Model
Constraints
      ↓
 Portfolio Optimizer
      ↓
 Portfolio Weights
```

Composers are the primary mechanism for expressing relationships between data sources.

---

## Characteristics

Every Composer should satisfy the following properties.

### Multiple Inputs

A Composer accepts one or more Panels.

```text
Panel A
Panel B
Panel C
    ↓
 Composer
    ↓
 Output Panel
```

---

### Single Output

A Composer produces exactly one output Panel.

```text
(P₁, P₂, ..., Pₙ) → P
```

---

### Graph Node

A Composer is a first-class node within the DAG.

Example:

```text
Factor A
Factor B
    ↓
 Combine
    ↓
 Alpha
```

---

### Reusable

A Composer should be reusable across different workflows.

Example:

```text
Divide
```

can be used for:

```text
Book Value / Price
```

```text
Sales / Market Cap
```

```text
Cash Flow / Enterprise Value
```

---

## Examples

### Example 1: Divide

```text
Book Value
Price
     ↓
   Divide
     ↓
 BM Ratio
```

This Composer combines two Panels into a valuation signal.

---

### Example 2: Add

```text
Factor A
Factor B
     ↓
    Add
     ↓
Combined Factor
```

---

### Example 3: Weighted Sum

```text
Factor A
Factor B
Factor C
      ↓
 Weighted Sum
      ↓
 Alpha Signal
```

---

### Example 4: Linear Regression

```text
Factor A
Factor B
Factor C
      ↓
     OLS
      ↓
 Prediction
```

---

### Example 5: Neural Network

```text
BM Factor
ROE Factor
Momentum Factor
        ↓
  Neural Network
        ↓
    Prediction
```

---

### Example 6: Portfolio Optimization

```text
Prediction
Risk Model
Constraints
      ↓
 Optimizer
      ↓
 Portfolio Weights
```

---

## Composer Categories

Composers can be organized into several categories.

---

### Arithmetic Composers

Basic mathematical combinations.

Examples:

```text
Add
Subtract
Multiply
Divide
```

Example:

```text
Book Value
Price
     ↓
   Divide
```

---

### Aggregation Composers

Combine multiple Panels into a single signal.

Examples:

```text
Weighted Sum
Average
Median
Voting
```

Example:

```text
Factor A
Factor B
Factor C
      ↓
 Weighted Sum
      ↓
 Combined Alpha
```

---

### Statistical Model Composers

Fit statistical relationships.

Examples:

```text
OLS
Ridge
Lasso
PCA
```

Example:

```text
Factors
    ↓
  OLS
    ↓
Prediction
```

---

### Machine Learning Composers

Learn nonlinear relationships.

Examples:

```text
Random Forest
XGBoost
Neural Network
Transformer
```

Example:

```text
Features
    ↓
Neural Network
    ↓
Prediction
```

---

### Portfolio Construction Composers

Convert forecasts into portfolios.

Examples:

```text
Mean Variance
Risk Parity
Custom Optimizer
Black-Litterman
```

Example:

```text
Prediction
Risk Model
      ↓
 Optimizer
      ↓
 Portfolio
```

---

## Composer Chains

Composers may appear anywhere in a graph.

Example:

```text
Book Value
Price
     ↓
   Divide
     ↓
 BM Ratio
     ↓
  ZScore
     ↓
 BM Factor
```

Here:

```text
Divide
```

is a Composer.

```text
ZScore
```

is an Transformer.

---

Another example:

```text
BM Factor
ROE Factor
Momentum Factor
        ↓
 Neural Network
        ↓
 Prediction
        ↓
 Normalize
        ↓
 Portfolio Weights
```

Here:

```text
Neural Network
```

is a Composer.

```text
Normalize
```

is an Transformer.

---

## Composer vs Transformer

A common design question is:

> Should this be an Transformer or a Composer?

Rule:

### Transformer

Uses one input Panel.

```text
P → P
```

Examples:

```text
Rank
ZScore
Rolling Mean
Normalize
```

---

### Composer

Uses multiple input Panels.

```text
(P₁, P₂, ..., Pₙ) → P
```

Examples:

```text
Add
Divide
OLS
Neural Network
Optimizer
```

---

## Dependency Behavior

Composers create dependency relationships within the graph.

Example:

```text
Factor A
Factor B
      ↓
 Weighted Sum
      ↓
 Combined Factor
```

Dependency graph:

```text
Factor A ─┐
          ├─> Combined Factor
Factor B ─┘
```

If either input changes:

```text
Factor A Updated
```

the Composer must be recomputed.

---

## Trainable Composers

Some Composers contain trainable parameters.

Example:

```text
Features
    ↓
Neural Network
    ↓
Prediction
```

The neural network may contain:

```text
Weights
Biases
Hyperparameters
```

These parameters are considered part of the Composer configuration.

From the graph perspective:

```text
Composer
```

still behaves as:

```text
(P₁, P₂, ..., Pₙ) → P
```

---

## Design Guidelines

When creating a new Composer:

### Combine Information

A Composer should express relationships between Panels.

---

### Produce One Output

Even complex models should produce a single output Panel.

---

### Remain Modular

Prefer:

```text
Factor A
Factor B
    ↓
Weighted Sum
```

over:

```text
MegaAlphaGenerator
```

---

### Preserve Panel Semantics

Input:

```text
Panels
```

Output:

```text
Panel
```

The Panel abstraction should never be broken.

---

## Examples Across the Investment Process

### Factor Construction

```text
Book Value
Price
     ↓
   Divide
     ↓
 BM Ratio
```

---

### Alpha Combination

```text
Value
Quality
Momentum
      ↓
 Weighted Sum
      ↓
 Alpha
```

---

### Prediction Modeling

```text
Factors
    ↓
 Neural Network
    ↓
Prediction
```

---

### Portfolio Construction

```text
Prediction
Risk Model
Constraints
      ↓
 Optimizer
      ↓
 Portfolio Weights
```

---

## Summary

Composers are multi-input Panel transformations.

```text
(P₁, P₂, ..., Pₙ) → P
```

They are responsible for combining information, modeling relationships, and constructing higher-level signals from multiple Panels.

Together with Panels and Transformers, Composers form the three fundamental primitives of BagelQuant's computational graph model.
