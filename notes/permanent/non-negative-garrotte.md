---
title: Non-negative Garrotte
date: 2024-10-12 00:00
modified: 2024-10-12 00:00
status: draft
---

**Non-negative Garrotte** is an approach the estimating co-efficients of a [Linear Model](linear-model.md) which starts with the least squares solutions and applies penalties based upon nonnegative weights that are determined by a quadratic program. Some weights are usually zero.

It minimizes the follow equation:

$$
\sum_{i=1}^{N} \left( y_i - \alpha - \sum_{j} c_j \hat{\beta_j} x_{ij} \right)^2 \quad \text{subject to} \quad c_j \geq 0, \quad \sum_j c_j \leq t.
$$

It was the inspiration behind [Lasso](lasso.md).

Similar to [Variable Subset Selection](../../../permanent/variable-subset-selection.md) but less sensitive to small pertubations in data, because penalise all co-efficients instead of co-efficients not selected.

The Lasso doesn't require explicit calculation of the least squares solution, so it can be used in cases where p>n, scenarios in which the non-negative garrote breaks down. However, because it applies an L_1 norm penalty, it can't be formulated as efficiently as a quadratic program.

The non-negative garrote may be preferable to the Lasso in scenarios where best subset selection would be appropriate, but you want to reduce the variance, so relatively large n compared to p.

"A drawback of the garotte is that its solution depends on both the sign and the magnitude of the OLS estimates. In overfit or highly correlated settings where the OLS estimates behave poorly, the garotte may suffer as a result. In contrast, the lasso avoids the explicit use of the OLS estimates." [^1]

[Regression Shrinkage and Selection via the Lasso](../reference/papers/regression-shrinkage-and-selection-via-the-lasso.md). Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267–288. https://www.jstor.org/stable/2346178
