---
title: Matrix multiplication as composition
date: 2021-10-27 00:00
category: reference/videos
summary: Notes from [Matrix multiplication as composition](https://www.youtube.com/watch?v=) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series
status: draft
---

Notes from [Matrix multiplication as composition](https://www.youtube.com/watch?v=kYB8IZa5AuE) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series

For some problems, we want to describe the effects of applying one [[Matrix Transformation]] then another. For example, we may rotate then apply a shear.

We call this a "composition" of 2 transformations.

One way to think of this, is that we first apply the rotation, then apply the shear:

$\begin{bmatrix}1 && 1 \\ 0 && 1\end{bmatrix} \left( \begin{bmatrix}0 && -1 \\ 1 && 0\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \right)$

Turns out to be the same as taking the product of the 2 [[Matrix Transformation]], then applying that to a vector.

We can think of multiplying two matrices like applying one transformation then another.

We read transformations from right to left, mirroring function notation:

$f(g(x))$ - we first apply g to x, then f to that result.
$\begin{bmatrix}1 && 1 \\ 0 && 1\end{bmatrix} \begin{bmatrix}0 && -1 \\ 1 && 0\end{bmatrix}$ - apply the right matrix, then the left.

Another example:

$\overbrace{\begin{bmatrix}0 && 2 \\ 1 && 0\end{bmatrix}}^{M2} \overbrace{\begin{bmatrix}1 && -2 \\ 1 && 0\end{bmatrix}}^{M1}$

We can first multiple 
