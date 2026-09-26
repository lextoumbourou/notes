---
title: Huber Loss
date: 2025-05-05 00:00
modified: 2026-09-26 08:55
status: draft
---

**Huber Loss** is a [Loss Function](loss-function.md) for regression that combines [Mean-Squared Error](mean-squared-error.md) and [Mean Absolute Error](mean-absolute-error.md). For small errors (below a threshold $\delta$) it's quadratic like MSE, and for large errors it's linear like MAE:

$$
L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta \left(|y - \hat{y}| - \frac{1}{2}\delta\right) & \text{otherwise} \end{cases}
$$

This makes it less sensitive to outliers than MSE, while still being smooth around zero.

It's commonly used in [Deep-Q Learning](deep-q-learning.md) to stop large errors from causing huge, unstable updates.
