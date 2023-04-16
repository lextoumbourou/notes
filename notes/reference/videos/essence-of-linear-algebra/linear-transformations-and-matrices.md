---
title: Linear Transformations and Matrices
date: 2021-10-26 00:00
category: reference/videos
summary: Notes from [Linear transformations and matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series
cover: /_media/cover-linear-transformations.png
status: draft
parent: essence-of-linear-algebra
---

Notes from [Linear transformations and matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series

Breaking down "Linear Transformation":

* Transform: another name for a function, in that it takes an input and returns the output.
* However, a *"transformation"* implies the ability to move the transformation as *"movement"*.
* Linear: limits the types of transformation to ones that have these visual properties:
    * All lines, remain lines.
    * Origin remains fixed in place.
* This keeps the grid lines parallel and evenly spaced.

To record a transformation numerically, you only need to store where the [Basis Vectors](../../../permanent/basis-vectors.md), $\hat{i}$ and $\hat{j}$ land.

Consider vector $\vec{v} = \begin{bmatrix}-1 \\ 2\end{bmatrix}$

We know it's just a linear combination of $\hat{i}$ and $\hat{j}$: $\vec{v} = -1\hat{i} + 2\hat{j}$

We know grid lines become evenly spaced.

So, after applying the transformation, the property: $\vec{v} = -1\hat{i} + 2\hat{j}$ still applies, it's just the basis vectors that have been transformed.

That means we can fully describe a 2d transformation with four numbers (2 vectors).

We put these numbers into a 2x2 matrix, called a [Matrix Transformation](../../../permanent/matrix-transformation.md). Each column is one of the transformed [Basis Vectors](../../../permanent/basis-vectors.md)[Transformation matrix](../../../_media/linear-trans-transformation-matrix.png)

So for any vector, we can multiply each coordinate by the corresponding column of the transformation matrix to get the result:

$\begin{bmatrix}3 && 2 \\ -2 && 1\end{bmatrix} \begin{bmatrix}5 \\ 7\end{bmatrix} = 5\begin{bmatrix}3 \\ -2\end{bmatrix} + 7\begin{bmatrix}2 \\ 1\end{bmatrix}$

Another example is a 90 degree rotation of space. $\hat{i}$ ends up at $\begin{bmatrix}0 \\ 1\end{bmatrix}$, $\hat{j}$ ends up at $\begin{bmatrix}-1 \\ 0\end{bmatrix}$

We get transformation matrix: $\begin{bmatrix}0 && -1 \\ 1 && 0\end{bmatrix}$

![90 degree rotation](/_media/linear-trans-90-degree-rot.png)

We can determine where vector $\vec{u} = \begin{bmatrix}a \\ b\end{bmatrix}$ lands as $a\begin{bmatrix}0 \\ 1\end{bmatrix} + b\begin{bmatrix}-1 \\ 1\end{bmatrix}$

Another example is a [Sheer Transformation](../../../permanent/lintrans-shear.md). $\hat{i}$ remains fixed. $\hat{j}$ moves to the coordinates $\begin{bmatrix}1 \\ 1\end{bmatrix}$

We get transformation matrix: $\begin{bmatrix}1 && 1 \\ 0 && 1\end{bmatrix}$

![Shear transformation](/_media/linear-trans-shear-trans.png)

We can determine where vector $\vec{u} = \begin{bmatrix}a \\ b\end{bmatrix}$ lands as $a\begin{bmatrix}1 \\ 0\end{bmatrix} + b\begin{bmatrix}1 \\ 1\end{bmatrix}$

If the columns of a transformation matrix are linearly dependent, the linear transformation "squishes" space into a single line.

![Linear dependent columns](/_media/linear-trans-linear-dep-columns.png)

The span of those 2 vectors is 1-dimensional in this case.
