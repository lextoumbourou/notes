---
title: "Week 5 - What are eigenvalues and eigenvectors?"
date: 2021-10-04 11:30
category: reference/moocs
status: draft
parent: linear-algebra-machine-learning
tags:
  - LinearAlgebra
  - MachineLearningMath
  - GameMath
---

## What are eigen-things?

### What are eigenvalues and eigenvectors?

* "Eigen" is translated from German as "characteristic."
* "Eigenproblem" is about finding characteristic properties of something.
 
* Geometric interpretation of [[Eigenvectors]] and [[Eigenvalues]] (00:45-04:22)
    * Though we typically visualize linear transformations based on how they affect a single vector, we can also consider how they affect every vector in the space by drawing a square.

        ![Visualing all vectors in space as square](/_media/laml-all-vectors-in-space-square.png)
        
        We can then see how a transformation distorts the shape of the square.

    * If we apply a scaling of 2 in vertical direction $\left( \begin{bmatrix}1 & 0 \\ 0 & 2\end{bmatrix} \right)$ the square becomes a rectangle.

        ![Vertical scaling](/_media/laml-vertical-scaling-to-rectangle.png)

    * A horizontal sheer on the other hand looks like this: $\left( \begin{bmatrix}1 & s \\ 0 & 1\end{bmatrix} \right)$

        ![Horizontal sheer transform](/_media/laml-horizontal-sheer-transform.png)

    * When we perform these operations:
        * Some vectors point in the same direction but change length.
        * Some vectors point in a new direction.
        * Some vectors do not change.

        ![Highlighted vectors after scaling](_media/laml-vectors-after-scaling.png)

    * The vectors that point in the same direction we refer to as [[Eigenvectors]].
    * The vectors that point in the same direction and whose size does not change are said to have [[Eigenvalues]] 1.
        * In the above example, the vertical eigenvector doubles in length, with an Eigenvalue of 2.

    * In a pure sheer operation, only the horizontal vector is unchanged. So the transformation has 1 Eigenvector.
    * In a rotation, there are no Eigenvectors.
