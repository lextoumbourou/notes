---
title: Matrix multiplication as composition
date: 2021-11-02 00:00
category: reference/videos
summary: Notes from [Matrix multiplication as composition](https://www.youtube.com/watch?v=) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series
cover: /_media/matrix-mult-as-composition-cover.png
status: draft
parent: essence-of-linear-algebra
---

Notes from [Matrix multiplication as composition](https://www.youtube.com/watch?v=kYB8IZa5AuE) by 3Blue1Brown from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series

Sometimes we want to describe the effects of applying one [Matrix Transformation](../../../permanent/matrix-transformation.md) than another. For example, we may rotate then apply a shear.

We call this a "composition" of 2 transformations.

One way to think of this, is that we first apply the rotation, then apply the shear:

$\begin{bmatrix}1 && 1 \\ 0 && 1\end{bmatrix} \left( \begin{bmatrix}0 && -1 \\ 1 && 0\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \right)$

It turns out to be the same as taking the 2 [Matrix Transformation](../../../permanent/matrix-transformation.md) product, then applying that to a vector.

We can think of multiplying two matrices, like applying one transformation then another.

We read transformations from right to left, mirroring function notation. For example, $f(g(x))$ we first apply g to x, then f to that result.

$\begin{bmatrix}1 && 1 \\ 0 && 1\end{bmatrix} \begin{bmatrix}0 && -1 \\ 1 && 0\end{bmatrix}$ - apply the right matrix, then the left.

Another example:

$\overbrace{\begin{bmatrix}0 && 2 \\ 1 && 0\end{bmatrix}}^{M2} \overbrace{\begin{bmatrix}1 && -2 \\ 1 && 0\end{bmatrix}}^{M1}$

The total effect of applying $M1$ and then $M2$ gives us a new matrix: $\begin{bmatrix}2 && 0 \\ 1 && -2\end{bmatrix}$

Note that [Matrix Multiplication](../../../permanent/matrix-multiplication.md) is not *commutative*: $M1 \ne M2$. Order matters.

It is however *associative*: $(AB)C = A(BC)$
