---
title: Combinatorics
date: 2023-04-11 00:00
status: draft
summary: A system for counting the number of ways a task can be achieved.
---

Combinatorics is a branch of math concerned with collections of arrangement of finite objects. It helps us answer the question of how many ways a task can be achieved.

What do we mean by "task"?

Here's some example:

* How many ways can you draw a black queen or a red king from a deck of cards?

In this example, drawing a black queen we'll call task and a red king is task B.

Since there are 2 ways to draw a black queen, we would say A = 2 and same for task B, so B = 2.

## [Addition Principle](combinatorics-addition-principle.md)

Suppose A can be done in $m$ ways and B can be done in $n$ ways, if A and B are mutually exclusive tasks, as they are in the above example, then task A or task B can be achieved in m + n ways.

## [Inclusion-Exclusion Principal](inclusion-exclusion-principal.md)

If we have a set of tasks A that can achieved in $m$ different ways, and task B that can be achieved in $n$ different ways, and also a number of ways $k$ that both can be accomplished, then task $A$ or task $B$ can be achieved in $m + n - k$ different ways.

## [Multiplication Principle](multiplication-principle.md)

There are 2 key ideas: [Permutation](permutation.md)s and [Combination](Combination)s.
