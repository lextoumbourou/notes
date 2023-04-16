---
title: Eigenvectors and Eigenvalues
date: 2022-01-08 00:00
category: reference/videos
cover: /_media/3blue-eigen-cover.png
summary: "Notes from [Eigenvectors and eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g) by Khan Academy."
hide_cover_in_article: true
status: draft
parent: essence-of-linear-algebra
---

These are notes from [Eigenvectors and eigenvalues | Chapter 14, Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g) by Khan Academy.

Consider a linear transformation in 2d, $\begin{bmatrix}3 && 1 \\ 0 && 2\end{bmatrix}$. Most vectors don't continue along their span after the transformation.

![Vector knocked off span](/_media/3blue-vector-knocked.png)

However, some particular vectors do remain on their span. The vector $\hat{i}$ moves over to 3x itself but still on the x-axis.

![Vector remain on span after transformation](/_media/3blue-vector-remain-on-span.png)

Because of how linear transformations work, any other vector on the same axis is scaled by 3.

Another vector $\begin{bmatrix}-1 \\ 1\end{bmatrix}$ also stays on its span during the transformation. It is stretched by a factor of 2, as does any vector on the same span.

![Eigenvectors stay on the same span](/_media/3blue-eigenvectors.png)

These vectors are called the [Eigenvector](../../../permanent/eigenvector.md)s.

Each Eigenvector has an associated [Eigenvalue](../../../permanent/eigenvalue.md). The Eigenvalue refers to how it stretches or squishes the Eigenvector during the transformation.

The Eigenvalues can be negative and fractional.

Finding Eigenvectors is useful in 3d transformations because the Eigenvector is the axis of rotation.

![Eigenvectors in 3d transformation](/_media/3blue-eigenvectors-axis-of-rotation.png)

It is much easier to think about a 3d rotation using the axis of rotation and the angle it's rotating, rather than thinking of the 3x3 matrix that describes the rotation.

The Eigenvalue here must be one since rotations don't stretch or squish anything.

Sometimes a way to understand a transformation is to find the Eigenvectors and Eigenvalues.

Symbolically, an Eigenvector is described like this:

$A\vec{v} = \lambda\vec{v}$

Where:
* $A$ is the matrix representing a transformation.
* $\vec{v}$ is the Eigenvector.
* $\lambda$ is a number representing the Eigenvalue.

The expression is saying: the Matrix-vector product $A\vec{v}$ is the same as scaling the Vector by $\lambda$.

So finding the Eigenvalues and Eigenvectors of $A\vec{v}$ is about finding the value $\lambda$ and $\vec{v}$ that make the expression true.

It's a bit strange to have matrix multiplication on one side and scalar multiplication on the other side, so we can rewrite it using the identity matrix as follows: $A\vec{v} = \left(\lambda I\right)\vec{v}$

We can then subtract off the right-hand side: $A\vec{v} - (\lambda I)\vec{v} = \vec{0}$

Then factor out the $\vec{v}$: $(A - \lambda I)\vec{v} = \vec{0}$

If $\vec{v}$ is 0, that will satisfy the answer, but you want a non-zero $\vec{v}$ as an Eigenvector.

From the lesson on [Matrix Determinate](../../../permanent/matrix-determinate.md), we know that the only way the transformation of a matrix with a non-zero vector is if transformation associated with that matrix squishes space onto a lower dimension. That corresponds to a 0 determinate for the matrix.

You can think about the determinate of this matrix: $\det\left(\begin{bmatrix}2 && 2 \\ 1 && 3\end{bmatrix}\right) = 4$

If you had a $\lambda$ value that was subtracting off each diagonal entry: $\det\left(\begin{bmatrix}2 - \lambda && 2 \\ 1 && 3 - \lambda\end{bmatrix}\right) = 4$

What lambda value would you need to get a 0 determinate? In this case, it happens when $\lambda = 1$

$\det\left(\begin{bmatrix}2 - 1 && 2 \\ 1 && 3 - 1\end{bmatrix}\right) = 0$

 So there's some vector when multiplied by $(A - \lambda I)$ equals 0.

 ---

 Another example: to find out if a value is an Eigenvalue, subtract it from the diagonal and compute the determinate.

 $\det\left(\begin{bmatrix}3 - \lambda && 1 \\ 0 && 2 - \lambda\end{bmatrix}\right)$

 Doing that, gives you a quadratic polynomial in $\lambda$: $\left(3 - \lambda\right)\left(2 - \lambda\right) = 0$

 We know that the $\lambda$ can only be an Eigenvalue if the determinate is 0; we can conclude that the only possible values for $\lambda$ are 3 and 2.

 To find the *Eigenvectors* that have these values, plug in the value to the matrix and solve to find which vectors return 0.

 $\begin{bmatrix}3 - 2 && 1 \\ 0 && 2 - 2\end{bmatrix} \begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}0 \\ 0\end{bmatrix}$

 Some 2d transformations have no Eigenvectors (what about 3d transformations?). For example, a rotation transformation $\begin{bmatrix}0 && -1 \\ 1 && 0\end{bmatrix}$ takes all vectors of their span.

 If we try to calculate the determinate: $(-\lambda)(-\lambda) - (-1)(1)$, the only possible solution is for $\lambda$ to be imaginary number $i$ or $-i$. Having no real number solutions tells us there aren't any Eigenvectors.

 A shear is another interesting example. It fixes $\hat{i}$ in place and makes $\hat{j} = \begin{bmatrix}1 \\ 1\end{bmatrix}$. All vectors on x-axis are Eigenvectors with Eigenvalues of 1.

 When you calc the determinate, $(1-\lambda)(1-\lambda) - 1 \cdot 0$, you get $1 - \lambda^2$. The only root of the expression is $1$.

 It's possible to have one Eigenvalue with multiple Eigenvectors. A matrix that scales everything by 2: $\begin{bmatrix}2 && 0 \\ 0 && 2\end{bmatrix}$ makes every vector in the plane an Eigenvector, with the only Eigenvalue being 2.

What happens if both basis vectors are Eigenvectors? One example is $\begin{bmatrix}-1 && 0 \\0 && 2\end{bmatrix}$.

 Notice how there's a positive value on the diagonal and 0s everywhere else? That's a [Diagonal Matrix](../../../permanent/diagonal-matrix.md).

 The way to interpret it is that all the basis vectors are Eigenvectors, with the diagonal entry being Eigenvalues.

 There are a lot of things that make diagonal matrices much easier. One of them is that it's easy to reason about what happens when you apply the matrix multiple times. You are simply multiplying the diagonal values multiple times.

 Contrast that with normal matrix multiplication. It quickly gets complicated.

 The basis vectors are rarely Eigenvectors. But if your transformation has at least 2 Eigenvectors that span space, you can change your coordinate system so that your Eigenvectors are your basis vectors by [Changing Basis](../../../permanent/changing-basis.md). The composed matrix will be Diagonal Matrix.

 So, if you need to compute the 100th power of a matrix, it's easier to first convert to an Eigenbasis. Perform the computation. Then convert back to the original basis.

 Note that not all transformations will support this. Shear or rotation don't have enough Eigenvectors to support this, for example.
