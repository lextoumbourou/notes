---
title: PageRank
date: 2026-09-26 09:05
modified: 2026-09-26 09:05
status: draft
---

**PageRank** is an algorithm, developed by Larry Page and Sergey Brin at Google, that ranks web pages by importance based on the importance of the pages that link to them.

The web is modelled as a graph, and each page's outgoing links become a column of a link matrix, normalised so they sum to 1. The ranks are the [Eigenvector](eigenvector.md) of this matrix with an [Eigenvalue](eigenvalue.md) of 1. You can think of it as the long-run probability that a random surfer clicking links ends up on each page.

Because the matrix for the whole web is huge and mostly zeros, it's usually solved with the power method: start with a guess vector and repeatedly multiply it by the matrix until it stops changing. A damping factor is also added to model the surfer occasionally jumping to a random page.

See [Week 5 - What are eigenvalues and eigenvectors?](../reference/moocs/coursera/linear-algebra-machine-learning/week-5.md).
