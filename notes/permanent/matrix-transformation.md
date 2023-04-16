---
title: Matrix Transformation
date: 2021-11-01 00:00
tags:
  - LinearAlgebra
cover: /_media/transformation-matrix-cover.png
summary: A matrix as a transformation of a space.
---

We can think of a matrix as a transformation of a [Vector](vector.md) or all vectors in space.

When we take the product of a matrix and a vector, we are *transforming* the vector.

A transformation is another word for a function: it takes in some inputs (a vector) and returns some output (a transformed vector).

For example, we can rotate a vector $\begin{bmatrix}x \\ y\end{bmatrix}$ some angle $\theta$ about the origin using a [Rotational Matrix](rotational-matrix.md).

$\begin{bmatrix}\text{x*} \\ \text{y*} \end{bmatrix} = \begin{bmatrix}\cos\theta && \sin\theta \\ -\sin\theta && \cos\theta\end{bmatrix} \begin{bmatrix}x \\ y\end{bmatrix}$

In this example, we perform a 90° rotation of vector $\begin{bmatrix}2 \\ 2\end{bmatrix}$.

![Rotation matrix product with vector](/_media/transformation-matrix-example.gif)

To describe a transformation as a matrix, we only need to record where the [Basis Vectors](Basis Vectors.md) land as columns of a new matrix: $\begin{bmatrix}\color{red}{a} && \color{green}{b} \\ \color{red}{c} && \color{green}{d}\end{bmatrix}$

For example, a [Shear Transformation](Shear Transformation) keeps the $\hat{i}$ basis vector fixed, and slants the $\hat{j}$ basis vector. We can record that as: $\begin{bmatrix}\color{red}{1} && \color{green}{1} \\ \color{red}{0} && \color{green}{2}\end{bmatrix}$

![Transformed basis vectors](/_media/trans-basis.gif)

A matrix transformation is always linear in that it keeps all gridlines in space are parallel and evenly spaced.

[@3blue1brownVectorsChapter3Essence2016]

[@dyeMathematicsMachineLearning]

Image processing is a use case for matrix transformations.

Since we represent an image as a $m \ x \ n$ grid of pixels, we can treat the position of each pixel as a vector, then perform a transform of each position vector to transform the entire image.

In [this example](https://www.kaggle.com/lextoumbourou/image-rotation), I rotate an image using the rotational matrix above.

There's some additional code required:

* Create a new matrix that's the maximum possible width and height.
* Convert each position into a vector.
* Convert each position vector, so it's a distance from the center, not top-left.
* Rotate each position.
* Revert convert using a new image size.

{% notebook permanent/notebooks/rotation-mnist.ipynb cells[1:2] %}

Note that we end up with some empty pixels in a 45° rotation. These occur because some of the transformed coordinates are floating-point numbers. When they get rounded into integer positions, some of the pixels get excluded. There are many strategies to deal with this, but that's for another article.

[@agrawalRotatingImage]
