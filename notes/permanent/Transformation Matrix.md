---
title: Transformation Matrix
date: 2021-10-27 00:00
tags:
  - LinearAlgebra
summary: A special type of matrix for transforming vectors
status: draft
---

A transformation matrix is a special type of matrix used for transforming vectors. 

In 2d space, it's a 2x2 matrix where each column represents where the [[Basis Vectors]] "lands" [@3blue2brown].

For example, a 90° rotation takes the standard basis vectors from $\hat{i} = \begin{bmatrix}1 \\ 0\end{bmatrix}$, $\hat{j} = \begin{bmatrix}0 \\ 1\end{bmatrix}$ to $\hat{i} = \begin{bmatrix}1 \\ 0\end{bmatrix}$, $\hat{j} = \begin{bmatrix}-1 \\ 0\end{bmatrix}$, can be described in a single transformation matrix:

$\begin{bmatrix}1 && -1 \\0 && 0\end{bmatrix}$

That can then be used to perform a 90° rotation of any vector, remember from [[Basis Vectors]], that each element of a vector can be considered a scalar of a basis

$\begin{bmatrix}1 && -1 \\0 && 0\end{bmatrix} \begin{bmatrix}1 \\ 2\end{bmatrix} = 1\begin{bmatrix}1 \\0\end{bmatrix} 2\begin{bmatrix}-1 \\ 0\end{bmatrix}$

A use case for this is rotating images using a rotational matrix.

That moves the basis vectors some angle, represented by:

$\hat{x} = \begin{bmatrix}\cos\theta \\ \sin\theta\end{bmatrix}$, $\hat{y} = \begin{bmatrix}-\sin\theta \\ \cos\theta\end{bmatrix}$

Since an image is represented as a $m \ x \ n$ grid of pixels, we can treat the position of each pixel as a vector, then perform a transform of each position vector, rotation the entire image.

But first we:
* Create a new matrix that's the maximum possible width and height.
* Create a matrix of positions.
* Normalise each position so it's a distance from centre.
* Rotate each position.
* Revert normalistion using a new image size.
