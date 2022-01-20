---
title: Eigenvector
date: 2022-01-13 00:00
status: draft
tags:
  - LinearAlgebra
summary: "the vectors that remain on their span after a transformation"
---

The Eigenvectors of a [[Matrix Transformation]] are any non-zero vector that remains on there [[Vector Span]] after being transformed.

That means that performing the transformation is equavilent to scaling the vector by some amount. The amount which the transformation scales the Eigenvector is called the [[Eigenvalue]].

For example, after transformation $\begin{bmatrix}2 && 1 \\ 0 && 2\end{bmatrix}$ most vectors are "knocked" off their span. However, $\vec{a}$ is simply scaled by 2.

<video controls loop><source src="/_media/eigenvector.mp4" type="video/mp4"></video>

One other special case of vector that remains on its span is the $\vec{v}=\begin{bmatrix}0\\0\end{bmatrix}$, but that's not an Eigenvector.

---

When performing 3d rotations, the Eigenvectors are particularly useful as they describe the axis of rotation.

---

The notation for describe the relationship between the matrix transformation of the vector, and same scaling quality equivalent:

$A\vec{v} = \lambda\vec{v}$