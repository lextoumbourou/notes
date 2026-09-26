---
title: Mean-Squared Error
date: 2024-04-01 00:00
modified: 2026-09-26 08:55
status: draft
summary: The average squared difference between predictions and labels
---


A metric for assessing the quality of regression models, where we square the difference between labels and predictions and take the average.

$MSE = \frac{1}{N} \sum\limits_{i=1}^{N} (y_i - \hat{y}_i)^2$

More popular than [Mean Absolute Error](mean-absolute-error.md) as it significantly increases large errors.

The derivative of the MSE loss function with respect to a specific weight $w_j$ is given by:

$∂MSE/∂w_j = -2/N \times Σ(y_i - ŷ_i) \times x_ij$