---
title: Series
date: 2023-09-02 00:00
modified: 2023-09-02 00:00
status: draft
---

A series is the sum of a [Sequence](sequence.md):

$S = \sum\limits^{\infty}_{n=1} a_n = a_1 + a_2 + a_3 + ...$

## Partial Sum

A partial sum is the sum of part of the sequence:

$S_3 = a_1 + a_2 + a_3$

Example:

The nth partial sum of the series $\sum\limits_{n=1}^{\infty} a_n$ is given by

$S_n = \frac{n^2 + 1}{n + 1}$

Find $a_7$

We know that $S_7 = a1 + a2 + a3 + a4 ... a_7$

So $a_7$ must be $S_7 - S_6$

$\frac{7^2 + 1}{7 + 1} - \frac{6^2 + 1}{6 + 1} = \frac{50}{8} - \frac{37}{7} = \frac{350}{56} - \frac{296}{56} = \frac{54}{56} = \frac{27}{28}$

## Infinite Series as Limit of Partial sums

$S_n = \frac{2n^3}{(n+1)(n+2)}$

Since S_n represents the partial sum up to sequence element n, we can think of it as the limit of partial sums as n approaches infinity:

$S = \lim_{n \rightarrow \infty} S_n$

$\frac{2n^3}{n^2 + 3n + 2}$

Since the numerator has the highest order, the limit should approach infinty. Therefore, we say the series diverges.
