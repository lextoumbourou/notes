---
title: Linear Recurrence
date: 2026-09-26 09:12
modified: 2026-09-26 09:12
status: draft
---

**Linear Recurrence** is a recurrence relation in which each term of a sequence is a linear function of earlier terms in the sequence.

There are two types:

* **Linear homogeneous recurrence**: $a_n = c_1 a_{n-1} + c_2 a_{n-2} + ... + c_k a_{n-k}$, where $c_1, ..., c_k \in \mathbb{R}$ and $k$ is the degree of the relation.
* **Linear non-homogeneous recurrence**: the same, plus an extra term $f(n)$ that depends only on $n$: $a_n = c_1 a_{n-1} + ... + c_k a_{n-k} + f(n)$.

For example, the [Towers of Hanoi](towers-of-hanoi.md) moves follow $a_n = 2a_{n-1} + 1$, which is a first-order non-homogeneous linear recurrence.

See [Week 12 - Recursion B](../reference/moocs/coursera/uol-discrete-mathematics/week-12-recursion-b.md).
