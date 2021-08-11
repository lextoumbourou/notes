---
title: Dot Product
date: 2021-08-11 21:30
tags:
  - LinearAlgebra
  - MachineLearningMath
---

A dot product is an operation between 2 equal-sized vectors that returns a single number.

It works like this: multiply each element in the first vector by the corresponding element in the 2nd vector, then take the sum of the results.

$$
A=\begin{bmatrix}a_1, ..., a_N\end{bmatrix} 
$$
$$
B=\begin{bmatrix}b_1, ..., b_N\end{bmatrix}
$$

$$
A \cdot B = \begin{bmatrix}a_1 \cdot b_1 + ... + a_N \cdot b_B\end{bmatrix}
$$

Simple Python example:

{% notebook permanent/notebooks/dot-product-examples.ipynb cells[0:2] %}

Numpy example:

{% notebook permanent/notebooks/dot-product-examples.ipynb cells[2:4] %}

The length (or magnitude) of a vector is the square root of the dot product of itself (which is also the vector squared):

$$
\text{Length of A} = \sqrt{A \cdot A} = \sqrt{A^{2}}
$$