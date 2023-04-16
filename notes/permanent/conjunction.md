---
title: Conjunction
date: 2022-11-23 00:00
tags:
  - Logic
  - DiscreteMath
status: draft
---

Given two [Proposition](proposition.md) $p$ and $q$, their conjunction is the proposition $p$ *and* $q$.

It math it's denoted as: $p \land q$.

Latex symbol: `\land`.

In programming, it would be the statement `condition AND condition2`.

Example:

p: The temperature is above 30℃.
q: The weather is humid.

Their conjunction is given by:

$p \land q$: The temperature is above 30℃ and the weather is humid.

Only when both $p$ and $q$ are true is the proposition $p \land q$ considered true.

Truth table:

| $p$ | $q$ | $p \land q$ |
| --- | --- | ----------- |
| T   | T   | T           |
| T   | F   | F           |
| F   | T   | F           |
| F   | F   | F           |
