---
title: Nonsquare matrices as transformations between dimensions
date: 2021-11-16 00:00
category: reference/videos
summary: Notes from [Nonsquare matrices as transformations between dimensions](https://www.youtube.com/watch?v=v8VSDg_WQlA) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series
cover: /_media/nonsquare-matrices-cover.png
status: draft
parent: essence-of-linear-algebra
---

Notes from [Nonsquare matrices as transformations between dimensions](https://www.youtube.com/watch?v=v8VSDg_WQlA) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series.

It's possible to have [Matrix Transformation](../../../permanent/matrix-transformation.md)s between dimensions, for example, one that takes 2d vectors as input and returns 3d vectors.

The transformation is linear as long as grid lines remain parallel and evenly spaced and the origin remains fixed.

Encoding a transformation as a matrix is the same as before: look where the [Basis Vectors](../../../permanent/basis-vectors.md)s land and record as columns of a matrix as a $3 \ x \ 2$ matrix.

$\begin{bmatrix}\textcolor{red}{2} && \textcolor{green}{0} \\ \textcolor{red}{-1} && \textcolor{green}{1} \\ \textcolor{red}{-2} && \textcolor{green}{1}\end{bmatrix}$

That transformation takes $\hat{i}$ to $\begin{bmatrix}\textcolor{red}{2} \\ \textcolor{red}{-1} \\ \textcolor{red}{-2}\end{bmatrix}$ and $\hat{j}$ to $\begin{bmatrix}\textcolor{green}{0} \\ \textcolor{green}{1} \\ \textcolor{green}{1}\end{bmatrix}$

The [Column Space](../../../permanent/column-space.md) (the place where all vectors land) of this matrix is a 2d plane slicing through the origin. However, we consider the matrix to be "full rank" since the number of dimensions in column space equals those in the input space.

So a $3 \ x \ 2$ matrix maps 2d space to 3d space: 2 columns indicate you're starting in space with two basis vectors.

On the other hand, a $2 \ x \ 3$ matrix maps 3d space to 3d space: 3 columns indicate you're starting in space with three basis vectors, two rows indicate that we describe the landing place with two coordinates.

We can also have a transformation from 2d to 1d (the number line). This transformation takes in vectors and returns numbers.

To create a visual understanding of what's happening in this case, create a line in 2d space of evenly spaced dots. After the transformation, the dots remain evenly spaced on the number line.
