---
title: Combination
date: 2023-04-11 00:00
status: draft
summary: an unordered collection of objects
tags:
    - DiscreteMath
    - Combinatorics
---

Unlike [Permutation](permutation.md), combinations are unordered set.

An r-combination of elements of a set is an unordered selection of r elements from the set.

The number of r-combinations of a set with n distinct elements is denoted by $C(n, r) = \binom{n}{r}$

The notation used is called **binomical coefficient**

Number of combinations
    * The number of r-combinations of a set with n distinct elements can be formulated as:
        * $C(n, r) = \frac{n!}{(n-r)!r!} = \frac{P(n, r)}{r!}$
    * $C(n, r)$ can be referred to as n choose r
    * It follows that $C(n, r) = C(n, n - r)$
