---
title: Cardinality
date: 2023-04-08 00:00
tags:
  - SetTheory
  - MachineLearning
summary: Cardinality refers to the number of elements a set contains.
cover: /_media/cardinality-cover.png
modified: 2023-04-13 00:00
---

The cardinality of a [Set](set.md) refers to the number of elements it contains.

In math notation, we represent the cardinality of a set $S$ as $|S|$. For example, the set $S = \{1, 2, 3\}$ has a cardinality of 3, expressed as $|S| = 3$.

In machine learning, the *"cardinality of a feature"* denotes the number of unique elements or categories within that feature. High-cardinality features may require feature engineering or be excluded entirely (for example, `user_id`).

<div clear="both"></div>

## Use of the vertical bar `|A|` notation

Initially it seemed confusing to me that mathematical notation employs the vertical bar symbol for different purposes.

For instance:

* The absolute value of a number $a$ is expressed as $|a|$.
* The [Matrix Determinate](matrix-determinate.md) of a matrix $\mathbf{M}$ is expressed as $|\mathbf{M}|$:

    $$
    \begin{aligned}
    \mathbf{M} = \begin{bmatrix}
    A & B \\
    C & D
    \end{bmatrix}
    ,\quad
    |\mathbf{M}| = AD - BC
    \end{aligned}
    $$

However, these notations share a common theme of representing size or magnitude:

* In set theory, cardinality describes the size of a set by the number of elements it contains.
* For numbers, the absolute value captures the distance from zero on the number line.
* In linear algebra, the [Matrix Determinate](matrix-determinate.md) determinant describes how much a [Matrix Transformation](matrix-transformation.md) scales space.
