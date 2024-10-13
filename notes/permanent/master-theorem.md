---
title: Master Theorem
date: 2024-02-24 00:00
modified: 2024-02-24 00:00
status: draft
---

The **Master Theorem** is a method for analysing the [Time Complexity](time-complexity.md) of recursive algorithms that follow a divide-and-conquer approach.

For the recurrence $T(n) = aT(\frac{n}{b}) + O(n^d)$ where $a \ge 1, b \gt 1, d \ge 0$

* $d < \log^{a}_{b} \rightarrow T(n) = O(n^{log^{a}_{b}})$
* $d = log^{a}_{b} \rightarrow T(n) = O(n^d \log n)$
* $d > \log^{a}_{b} \rightarrow T(n) = O(n^d)$

Example:

$T(n) = T(\frac{n}{2}) + O(1), a = 1, b = 2, d = 0$
$log^1_2 = 0$
$d = \log^{1}_{2}$
$T(n) = O(n^d \log n) = O(1 \log n) = O(\log n)$

Example:

$T(n) = 3T(\frac{n}{4}) + O(n), a = 3, b = 4, d = 1$
$1 > \log^{3}_{4}$
 $T(n) = O(n^d)$ so $T(n) = O(n)$
