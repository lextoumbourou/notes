---
title: Polynomial
date: 2026-07-23 00:00
modified: 2026-07-23 09:16
status: hidden
---

A **polynomial** is a finite sum of terms formed from constants and variables raised only to non-negative integer powers.

These are polynomials:

- $2x^2 + 10x - 5$
- $10x^2 - 3x - 2$
- $x^2y + 3xy^2 - 1$

These are non-polynomial expressions:

- $10x^{-3}$ (negative exponent)
- $\frac{1}{x} + 2$ (variable in the denominator)
- $\sqrt{x} - 5x$ (fractional exponent)
- $2^x + 1$ (variable exponent)
- $\log(x) + 3$ (logarithm)
- $\sin(x) + x$ (trigonometric function)
- $|x| + 1$ (absolute value)

---

A [Polynomial Function](../../../permanent/polynomial-function.md) is the mapping produced by evaluating a polynomial expression. For example, the expression $2x^2 + 2x + 1$ defines the function $f(x) = 2x^2 + 2x + 1$.

---

[Polynomial Time](../../../permanent/polynomial-time.md) in [Complexity Analysis](complexity-analysis.md) describes an algorithm whose running time is *bounded* by $O(n^k)$ for some constant $k$ that > 1 - really meaning **polynomial bounded time**. So it includes constant, logarithmic, linear and other running times bounded above by a polynomial, but not running times that grow exponentially like: $\Theta(2^n)$