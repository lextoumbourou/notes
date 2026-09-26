---
title: Orthogonal Matrix
date: 2026-09-26 09:05
modified: 2026-09-26 09:05
status: draft
---

**Orthogonal Matrix** is a square matrix whose columns (and rows) form an [Orthonormal Basis Set](orthonormal-basis-set.md): every column has length 1 and is perpendicular to every other column.

Multiplying an orthogonal matrix by its transpose gives the [Identity Matrix](identity-matrix.md), $A^{T}A = I$, so its transpose is also its inverse: $A^{-1} = A^{T}$. That makes the inverse very cheap to compute.

The determinant of an orthogonal matrix is always 1 or -1, and the transformation it describes preserves lengths and angles (it's a rotation, a reflection or a combination of both).

See [Week 4 - Matrices make linear mappings](../reference/moocs/coursera/linear-algebra-machine-learning/week-4.md).
