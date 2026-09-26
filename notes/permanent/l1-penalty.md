---
title: L1 Penalty
date: 2024-10-12 00:00
modified: 2026-09-26 09:08
status: draft
---

The **L1 Penalty** is a regularisation term that adds the sum of the absolute values of a model's weights, scaled by a hyperparameter $\lambda$, to the loss: $\lambda \sum_i |w_i|$.

It pushes many weights to exactly zero, so it tends to produce sparse models. It's the penalty used in [Lasso](lasso.md) regression.

It's not the same as the L1 loss, which measures the absolute error of predictions. See [Mean Absolute Error](mean-absolute-error.md).
