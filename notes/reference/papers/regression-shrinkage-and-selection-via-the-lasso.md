---
title: Regression Shrinkage and Selection via the Lasso
date: 2024-10-12 00:00
modified: 2024-10-12 00:00
status: draft
tags:
- MachineLearning
- Interpretability
---

These are my notes from the paper [Regression Shrinkage and Selection via the Lasso](https://webdoc.agsci.colostate.edu/koontz/arec-econ535/papers/Tibshirani%20(JRSS-B%201996).pdf) by [Robert Tibshirani](https://en.wikipedia.org/wiki/Robert_Tibshirani).

## Overview

The paper introduces a method for estimating coefficients of a [Linear Regression](../../permanent/linear-regression.md) model called [Lasso](../../permanent/lasso.md) or **Least Absolute Shrinkage and Selection Operator**.

It builds on [Ordinary Least Squares](../../../../permanent/ordinary-least-squares.md) by ensuring the sum of the absolute values of the coefficients is below some value $t$.

$$
(\hat{\alpha}, \hat{\beta}) = \arg \min \left\{ \sum_{i=1}^{N} \left( y_i - \alpha - \sum_{j} \beta_j x_{ij} \right)^2 \right\} \quad \text{subject to} \quad \sum_{j} |\beta_j| \leq t.
$$

Unlike alternative methods [Variable Subset Selection](../../../../permanent/variable-subset-selection.md) it tends to generalise better, and [Ridge Regression](../../../../permanent/ridge-regression.md) tends to make more interpretable models, since it typically makes some coefficients exactly 0 and prioritises key features, making models more interpretable, i.e. we can understand which coefficients impact the outcome the most.

## Lasso Algorithm

Given data $(\mathbf{x}^{i}, y_i)$ for $i =1, 2, \ldots, N$, where $\mathbf{x}^i = (x_{i_1}, \ldots, x_{i_p})^T$ are the predictor variables, and $y_i$ are the labels.

The observations are assumed to be independent, or the $y_i$ are conditionally independent given the $x_{ij}$. Assumes predictors are standardised so that $\sum_{i}x_{ij}/N = 0$ and $\sum_{i}x_{ij}^2 / N = 1$.

The LASSO estimate $(\hat{\alpha}, \hat{\beta})$ is defined as:

$$
(\hat{\alpha}, \hat{\beta}) = \arg \min \left\{ \sum_{i=1}^{N} \left( y_i - \alpha - \sum_{j} \beta_j x_{ij} \right)^2 \right\} \quad \text{subject to} \quad \sum_{j} |\beta_j| \leq t.
$$

$t > 0$ is a tuning parameter.

For all $t$, the solution for $\alpha$ is $\hat{\alpha} = \bar{y}$, We can assume without loss of generality that $\hat{y} = 0$ and hence omit the bias term $\alpha$, focusing on the coefficients $\hat{\beta}$

Computation of the solution is a quadratic programming problem with linear inequality constraints.

The parameter $t \ge 0$ controls the amount of shrinkage applied to the estimates.

Let $\hat{\beta}^0_j$ be the full least squares estimates and let $t_0 = \sum_j |\hat{\beta}^0_j|$. Values of $t < t_0$ will cause shrinkage of the solutions towards 0, and some coefficients may be exactly equal to 0. For example, if $t = t_0/2$, the effect will be roughly similar to finding the best subset of size $p/2$. Note also that the design matrix does not need to be of full rank.

 The motivation for the Lasso came from Breiman's [Non-negative Garrotte](../../permanent/non-negative-garrotte.md), which minimises:

$$
\sum_{i=1}^{N} \left( y_i - \alpha - \sum_{j} c_j \hat{\beta_j} x_{ij} \right)^2 \quad \text{subject to} \quad c_j \geq 0, \quad \sum_j c_j \leq t.
$$

A drawback of the garotte is that depends on both the sign and the magnitude of the [Ordinary Least Squares](../../../../permanent/ordinary-least-squares.md) estimates.

In overfit or highly correlated settings where the OLS estimates behave poorly, the garotte may suffer as a result. In contrast, the Lasso avoids the explicit use of the OLS estimates.
