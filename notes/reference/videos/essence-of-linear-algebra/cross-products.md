---
title: Cross Products
date: 2021-12-05 00:00
category: reference/videos
summary: Notes from [Cross products | Chapter 10, Essence of linear algebra](https://www.youtube.com/watch?v=eu6i7WJeinw)) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series.
status: draft
parent: essence-of-linear-algebra
---

These are notes from [Cross products | Chapter 10, Essence of linear algebra](https://www.youtube.com/watch?v=eu6i7WJeinw)) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series.

Like the Dot Product introduction, this lesson will show the standard intro to Cross Products, then provide a deeper understanding.

If you have two vectors in 2-dimensional space, think about the parallelogram they create when you copy each vector and move them to the tip of the other vector.

The area of that parallelogram is part of the cross product. Although, when $\vec{v}$ is on the right, then $\vec{v} \times \vec{w}$ is positive, and on the left, then the area is negative.

![Cross product in 2d](/_media/3blue-2d-cross-product.png)

That means that the cross-product order matters. If you switched the order, the cross product would become the negative of what it was before.

$\vec{v} \times \vec{w} = -\vec{w} \times \vec{v}$

The way to remember is, if you cross $\hat{i}$ and $\hat{j}$ in order, the result will be positive. If not, it will be negative. For any vector first in the cross product, if it's on the right of the other vector, the result will be positive on the left, negative.

You can compute the area using the [Matrix Determinate](../../../permanent/matrix-determinate.md) of the matrix created by putting the coordinates of $\vec{v}$ as the first column and $\vec{w}$ as the second column. That's because a [Matrix Transformation](../../../permanent/matrix-transformation.md) equivalent of moving the [Basis Vectors](../../../permanent/basis-vectors.md) to $\vec{v}$ and $\vec{w}$ is this matrix!

If the vectors are perpendicular, the area will be greater than if the vectors are closer together.

If you scale one vector by 3, the determinate is also scaled by 3.

But this operation isn't technically the cross-product. The cross-product operation only exists in 3d space. The actual cross-product combines two different 3d vectors to create a new 3d vector.

$\vec{v} \times \vec{w} = \vec{p}$

The new 3d vector's length will be the area of the parallelogram defined by the 2d vectors. The direction of the new vector will be perpendicular to the parallelogram.

Using the "right-hand rule," we can determine which direction the new vector will be facing.

Point your index finger in the direction of $\vec{v}$ and middle finger in the direction of $\vec{w}$. When you stick out your thumb, that's the direction of the cross product.

![Right-hand rule](/_media/3blue-right-hand-rule.png)

For general computations, there is a formula you can memorise:

$\begin{bmatrix}v_1 \\ v_2 \\ v_3\end{bmatrix} \times \begin{bmatrix}w_1 \\ w_2 \\ w_3\end{bmatrix} = \begin{bmatrix}v_2 \cdot w_3 - w_2 \cdot v_3 \\ v_3 \cdot w_1 - w_3 \cdot v_1 \\ v_1 \cdot w_2 - w_1 \cdot v_2\end{bmatrix}$

But it's easier to remember an operation that involves a 3d determinant. Create a 3d matrix where the 2nd and 3rd column are $\vec{v}$ and $\vec{w}$:

$\begin{bmatrix} && v_1 && w_1 \\ && v_2 && w_2 \\ && v_3 && w_3\end{bmatrix}$

Then, the first column should contain the basis vectors $\hat{i}$, $\hat{j}$ and $\hat{k}$ (even though it is very weird to have basis vectors as entries of a matrix).

$\begin{bmatrix}\hat{i} && v_1 && w_1 \\ \hat{j} && v_2 && w_2 \\ \hat{k} && v_3 && w_3\end{bmatrix}$

You then compute the determinate of a 3d matrix, as if the first column was numbers.

$\det\left( \begin{bmatrix}\hat{i} && v_1 && w_1 \\ \hat{j} && v_2 && w_2 \\ \hat{k} && v_3 && w_3\end{bmatrix} \right)$

$= \hat{i}\left(v_2w_3 - v_3w_2\right) + \hat{j}\left(v_3w_1 - v_1w_3\right) + \hat{k}\left(v_1w_2 - v_2w_1\right)$

The multiplication on the right returns a number for each part of the expression. Then you scale up the basis vector by that number and take the combination of all scaled basis vectors.
