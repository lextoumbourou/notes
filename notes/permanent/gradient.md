---
title: Gradient
date: 2024-04-12 00:00
modified: 2026-09-26 08:50
status: draft
---

The **Gradient** is the generalisation of the [Derivative](derivative.md) for functions with multiple input variables.

For a function $f(x_1, \ldots, x_n)$, the gradient $\nabla f$ is the vector of its partial derivatives:

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n} \right)
$$

It points in the direction of steepest increase of the function, and its magnitude is how fast the function increases in that direction. That's why [Gradient Descent](gradient-descent.md) takes steps in the opposite direction to minimise a function.
