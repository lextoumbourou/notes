---
title: Cross Product
date: 2021-12-05 00:00
tags:
  - LinearAlgebra
summary: An operation between two 3d vectors that returns a vector.
cover: /_media/cross-product-cover.png
hide_cover_in_article: true
---

The Cross Product is an operation between two vectors that returns a vector: $\vec{a} \times \vec{b} = \vec{c}$

The returned vector will be orthogonal to both input vectors and have a length equal to the [Parallelogram Area](parallelogram-area.md) the input vectors define.

<video controls loop><source src="/_media/cross-product.mp4" type="video/mp4"></video>

Though the Cross Product can be generalized to multiple dimensions by taking the product of $n - 1$ vectors, you usually calculate the Cross Product in 3d space.

The Cross Product operation works as follows:

1. Line up the vectors, as you would when taking the [Dot Product](dot-product.md).

    $\begin{bmatrix}a_1 \\ a_2 \\ a_3\end{bmatrix} \times \begin{bmatrix}b_1 \\ b_2 \\ b_3\end{bmatrix} = \begin{bmatrix} ? \\ ? \\ ? \end{bmatrix}$

2. For the first component of the new vector, exclude the top rows of the input vectors. Then, calculate the 2d [Matrix Determinate](Matrix Determinate.md) of the matrix created by the bottom two rows of each matrix.

    $\det\left( \begin{bmatrix} \\ a_2 \\ a_3\end{bmatrix} \times \begin{bmatrix} \\ b_2 \\ b_3\end{bmatrix} \right) = \begin{bmatrix}\mathbf{a_2b_3 - a_3b_2} \\ \\ \end{bmatrix}$

3. For the 2nd component of the new vector, exclude the middle row of the input vectors. Then, we calculate the determinate; however, this time, you flip the order of the operations from $ad - bc$ to $bc - ad$.

    $\left( \begin{bmatrix} a_1 \\  \\ a_3\end{bmatrix} \times \begin{bmatrix} b_1 \\  \\ b_3\end{bmatrix} \right) = \begin{bmatrix}a_2b_3 - a_3b_2 \\ \mathbf{a_3b_1 - a_1b_3} \\ \ \end{bmatrix}$

4. Perform a 2d determinate operation, excluding the input vector's last rows for the final component.

    $\det\left( \begin{bmatrix} a_1 \\ a_2 \\ \ \end{bmatrix} \times \begin{bmatrix} b_1 \\ b_2 \\ \ \end{bmatrix} \right) = \begin{bmatrix}a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ \mathbf{a_1b_2 - a_2b_1} \end{bmatrix}$

For each pair of vectors, there will be two vectors that are perpendicular to both. To find out which direction the Cross Product's vector faces, use the right-hand rule. For $\vec{a} \times \vec{b}$, adjust you right-hand so you can place your index finger in the direction of $\vec{a}$ and your middle finger in the direction of $\vec{b}$. Whichever direction your thumb is pointing in is the direction of the cross product.

The cross product is helpful because it tells you if your vectors are parallel when the length of the vector returned by the Cross Product is 0.

[@khanacademylabCrossProductIntroduction]
[@3blue1brownCrossProductChapterEssence2016]
