---
title: Introduction to matrix inverses
date: 2021-09-25 13:30
status: draft
---

## Intro to matrix inverses

When you multiply a matrix by the [[Identity Matrix]], the original matrix is returned.

A 2x2 identity matrix is: $\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$ and so on.

In regular math, if you have $1 \times a = a$ and also $\frac{1}{a} \times a = 1$

So, is there a matrix where, if you have $A$ and multiply by $A^{-1}$ you get $I$? There is. It's called Inverse of A.

For a 2x2 matrix, you can calculate the inverse with the following algorithm:

$A = \begin{bmatrix}a & b \\ c & d\end{bmatrix}$

The inverse is:

$A^{-1} = \frac{1}{ad-bc} \begin{bmatrix}d & -b \\ -c & a\end{bmatrix}$

The  $ad-bc$ is the [[Matrix Determinate]], which is expressed as $|A| = ad-bc$

## Determining invertible matrices

A singular matrix is a matrix that cannot be inverted.

If the determinate is 0, then you cannot find the inverse, as you cannot divide by 0 in this expression:

$A^{-1} = \frac{1}{ad-bc} \begin{bmatrix}d & -b \\ -c & a\end{bmatrix}$

Which can be expressed in math:

$A^{-1} = \text{undefined } \text{iff } |A| = 0$

So if this expression is 0, the matrix isn't invertible

$|A| = ad-bc$

So that's when:

$ad = bc$

Or if the ratios are the same:

$\frac{a}{b} = \frac{c}{d}$

Or:

$\frac{a}{c} = \frac{b}{d}$

An example:

$\begin{bmatrix}a & b \\ c & d\end{bmatrix} \begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}e \\ f\end{bmatrix}$

Which  can be rewritten as:

$ax + by = e$

$cz + dy = f$

To solve for $y$:

$y = -\frac{a}{b}x + eb$

$y = -\frac{c}{d}x + \frac{f}{d}$

When there is no inverse, the lines for the 2 expressions are parallel: they never intersect.