---
title: Extended Euclidean Algorithm
date: 2025-03-15 00:00
modified: 2026-09-26 08:55
status: draft
---

**Extended Euclidean Algorithm** is an extension of the [Euclidean Algorithm](euclidean-algorithm.md) which, as well as finding the greatest common divisor of two integers $a$ and $b$, also finds integers $x$ and $y$ such that:

$ax + by = \gcd(a, b)$

This is known as Bézout's identity.

It's commonly used to find the modular multiplicative inverse of a number (see [Modular Arithmetic](modular-arithmetic.md)), which is a key step in generating keys for [RSA](rsa.md).
