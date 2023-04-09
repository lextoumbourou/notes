---
title: Basis Vectors
date: 2021-10-24 00:00
tags:
  - LinearAlgebra
cover: /_media/basis-vector-cover.png
summary: The set of vectors that defines space.
---

The set of [Vector](permanent/vector.md)s that defines space is called the *basis*.

 We refer to these vectors as *basis vectors*.

In 2d space, basis vectors are commonly defined as $\hat{i} = \begin{bmatrix}1 \\ 0\end{bmatrix}$ and $\hat{j} = \begin{bmatrix}0 \\ 1\end{bmatrix}$.

 These particular vectors are called the *standard basis vectors*. We can think of them as 1 in the direction of X and 1 in the direction of Y.

We can think of all other vectors in the space as [Linear Combinations](Linear Combinations) of the basis vectors.

For example, if I have vector $\begin{bmatrix}10 \\ -7\end{bmatrix}$, we can treat each component as scalar (see [Vector Scaling](permanent/Vector Scaling.md)) for the basis vectors: $10\begin{bmatrix}1 \\ 0\end{bmatrix} + (-7)\begin{bmatrix}0 \\ 1\end{bmatrix}$.

We can choose any set of vectors as the basis vectors for space, giving us entirely new coordinate systems. However, they must meet the following criteria:

* They're linear independent. That means you cannot get one [Vector](permanent/vector.md) by just scaling the other.
* They span the space. That means, by taking a linear combination of the two scaled vectors, you can return any vector.

Basis vectors don't have to be orthogonal to each other, but transformations become more challenging with a non-orthogonal basis.

In applications that require visualizing 3d space, like a game studio or a 3d modeling program, it's common for the basis vectors of the space to be displayed prominently. They are typically colored $\color{red}{X}\color{green}{Y}\color{blue}{Z}$, the color order mapping each axis to a color in RGB according to [this](https://ux.stackexchange.com/questions/79561/why-are-x-y-and-z-axes-represented-by-red-green-and-blue) post.

![Basis vectors in PlayCanvas](/_media/basis-vectors-playcanvas.png)

*Basis vectors in PlayCanvas*

[@3blue1brownVectorsChapter2Essence2016]
[@dyeMathematicsMachineLearning]