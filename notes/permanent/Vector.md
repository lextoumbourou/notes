---
title: Vector
date: 2020-11-08 18:00
modified: 2021-08-22 16:00
tags: 
  - LinearAlgebra 
  - GameMath
cover: /_media/vector.png
---

A vector is an ordered list of numbers.

We typically introduce vectors to math students as arrows pointing in space, defined by a length (or magnitude) and direction.

In this context, the numbers refer to coordinates in space that describe how far from the origin (in 2d space, that's usually the point `(0, 0)`) you travel to reach the tip of the vector. For a 2d vector, we can also describe the points as changes in x and y.

A vector can be moved around in space (in other words, we can change its origin), and it remains the same vector.

Vectors are represented in math as a vertical list of numbers, or a matrix with one column, to differential from coordinates.

$$\vec{A} = \begin{bmatrix}1\\2\end{bmatrix}$$

![Vector example](/_media/vector.png)

We can use Pythagoras Theorem to calculate the length of a vector by squaring each vector element, then taking the square root of their sum:

$$\| \vec{A} \| = \sqrt{1^2 + 2^2}$$

A computer scientist may think of vectors simply as an array. We can use vectors to represent a model of something. For example, the [Iris Flower Dataset](https://archive.ics.uci.edu/ml/datasets/iris) represents flowers as a vector of 4 attributes:

$$
\begin{bmatrix}
\text{sepal length}\\
\text{sepal width}\\
\text{petal length in cm}\\
\text{petal width in cm}
\end{bmatrix}
$$

We can plot these numbers in a coordinate system to see how the flowers relate to each other. This example from [Wikipedia](https://commons.wikimedia.org/wiki/File:Iris_dataset_scatterplot.svg) generates a 2d scatterplot for each pair of values.

![Iris scatterplot](/_media/iris-scatterplot.png)

* We can add or subtract vectors. See [[Vector Addition and Subtraction]]
* We can multiply vectors by a value. See [[Vector Scaling]]
* We can combine vectors in various ways. See [[Dot Product]] or [[Element-wise product]]

We describe a [[Ray]] with a similar notation to vectors. However, a Ray doesn't have a length. Only a direction to continue infinitely.

[@3blue1brownVectorsChapterEssence2016]