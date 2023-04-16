---
title: Matrix Multiplication
date: 2021-08-14 00:00
tags: 
  - LinearAlgebra 
---

Matrix multiplication is a mathematical operation between 2 matrices that returns a matrix.

For each row in the first matrix, take the [Dot Product](dot-product.md)f each column in the second matrix. Place the results onto the corresponding row in a new matrix.

![Matrix Multiplication example](/_media/matrix-multiplication.gif)

The size of the new matrix will be the row count from the first matrix by the column count from the second.

It is only defined when the number of columns from the first matrix equals the number of rows in the second.

In Numpy the `@` operator is used for matrix multiplication between 2 multi-dimensional arrays (matrices):

{% notebook permanent/notebooks/matrix-multiplication.ipynb cells[0:1] %}

The same operator works in PyTorch:

{% notebook permanent/notebooks/matrix-multiplication.ipynb cells[1:2] %}
