---
title: Vector Subtraction
date: 2021-01-06 00:00
cover: /_media/vector-subtract-cover.png
tags:
  - GameMath
  - LinearAlgebra
---

We subtract one [Vector](vector.md) from another by subtracting the corresponding components.

$$\vec{a} - \vec{b} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} - \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1 - b_1 \\ a_2 - b_2 \end{bmatrix} $$

Another way to think of vector subtraction, is [Vector Addition](Vector Addition.md) with the negative of a vector: $\vec{a} - \vec{b} = \vec{a} + (-\vec{b})$

We can visualize Vector subtraction as follows:

1. draw the 1st [Vector](vector.md)
2. draw the 2nd negative [Vector](vector.md) as a vector pointing in the opposite direction
3. Draw a line from the tail of the 1st [Vector](vector.md) to the tip of the negative 2nd [Vector](vector.md).

![Vector subtraction visual example](/_media/vector-subtract-example.gif)

Like [Vector Addition](Vector Addition.md), we can only subtract two vectors with the same number of dimensions.

[@3blue1brownVectorsChapterEssence2016]
