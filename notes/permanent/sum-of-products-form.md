---
title: Sum-of-Products Form
date: 2026-09-26 09:12
modified: 2026-09-26 09:12
status: draft
---

**Sum-of-Products Form** is a standard way of writing a [Boolean Function](boolean-function.md), where terms built with the AND operator (products) are combined with the OR operator (sum).

For example: $f(x, y, z) = xy + xz + yz$

Every Boolean function can be written in this form. To build it from a truth table, take each row where the function equals 1, write a product term where inputs equal to 1 appear uncomplemented and inputs equal to 0 appear complemented, then OR all of those terms together. The result isn't necessarily the simplest expression, so it's often simplified afterwards with [Boolean Algebra](boolean-algebra.md) theorems or a [Karnaugh Map](karnaugh-map.md).

The dual form is the [Product-of-Sums Form](product-of-sums-form.md). See [Week 9 - Boolean Algebra A](../reference/moocs/coursera/uol-discrete-mathematics/week-9-boolean-algebra-a.md).
