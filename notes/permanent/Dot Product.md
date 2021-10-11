---
title: Dot Product
date: 2021-08-11 21:30
tags:
  - LinearAlgebra
  - MachineLearningMath
---

A dot product is an operation between 2 equal-sized [[Vector]]s that returns a single number.

It works like this: multiply each element in the first [[Vector]] by the corresponding element in the 2nd [[Vector]], then take the sum of the results.

$$
\textbf{a} = \begin{bmatrix}a_1, ..., a_N\end{bmatrix}
$$
$$
\textbf{b} = \begin{bmatrix}b_1, ..., b_N\end{bmatrix}
$$

$$
\textbf{a} \cdot \textbf{b} = a_1 \cdot b_1 + ... + a_N \cdot b_N
$$

Simple Python example:

{% notebook permanent/notebooks/dot-product-examples.ipynb cells[0:2] %}

Numpy example:

{% notebook permanent/notebooks/dot-product-examples.ipynb cells[2:4] %}

The length (or magnitude) of a vector is the square root of the dot product of itself (which is also the vector squared):

$$
\text{Length of a} = \sqrt{\textbf{a} \cdot \textbf{a}}
$$

We can use the dot product to calculate the angle between 2 vectors. We know that the dot product of 2 vectors $r$ and $s$ is equal to the magnitude of the vectors * cosign of the angle between the vectors. $\|r\|\|s\|\cos\theta$. If we convert $\|r\|$ and $\|s\|$ into [[Unit Vector]]s, which are of size 1, then we are left with the cosign of the angle. We know this about cosigns:

cos(0°) = 1
cos(90°) = 1
cos(180°) = -1