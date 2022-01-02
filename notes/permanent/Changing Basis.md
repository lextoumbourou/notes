---
title: Changing Basis
date: 2021-12-13
tags:
  - LinearAlgebra
status: draft
---

The use of coordinates to define vectors implies an agreement about which [[Basis Vectors]] we use. In 2d space, we commonly use the standard basis vectors: $\hat{i}=\begin{bmatrix}1 \\ 0\end{bmatrix}$ and $\hat{j}=\begin{bmatrix}0 \\ 1\end{bmatrix}$. Any [[Vector]] that we describe in this basis we can think of as a linear combination of both basis vectors:

$\begin{bmatrix}a \\ b\end{bmatrix} = a\begin{bmatrix}1 \\ 0\end{bmatrix} + b\begin{bmatrix}0 \\ 1\end{bmatrix}$

However, any set of vectors can be used as a basis vectors. For example, a friend may use a grid system that uses basis vectors $\hat{e}_{1} = \begin{bmatrix}2 \\ 4\end{bmatrix}$ and $\hat{e}_{2} = \begin{bmatrix}1 \\ 1\end{bmatrix}$.

If our friend has a vector described in their coordinate system, for example $\begin{bmatrix}3 \\ 1\end{bmatrix}$, we can convert into our basis vectors by multiplying the vector by a matrix which has our friend's basis vectors as the columns:

$\begin{bmatrix}2 && 1 \\ 4 && 1\end{bmatrix}\begin{bmatrix}3 \\ 1\end{bmatrix} = \begin{bmatrix}7 \\ 13\end{bmatrix}$

We can convert a vector described in our coordinate system to our friend's using the [[Matrix Inverse]] of our friend's basis vector matrix:

$\begin{bmatrix}2 && 1 \\ 4 && 1\end{bmatrix}^{-1}\begin{bmatrix}7 \\ 13\end{bmatrix} = \begin{bmatrix}3 \\ 1\end{bmatrix}$

If we wish to perform a transformation described in our basis, for example, a rotational transformation in an alternate basis, we can follow these steps:

1. Convert the vector into our basis by applying our friend's transformation matrix.
2. Perform the translation.
3. Convert the vector back into our friend's basis by applying the inverse of the transformation.

In notation, if we have vector $\vec{v}$ described in our friend's basis $A$, we can apply [[Matrix Transformation]] $M$, described in our basis, as follows:

$A^{-1}MA \ \vec{v}$

