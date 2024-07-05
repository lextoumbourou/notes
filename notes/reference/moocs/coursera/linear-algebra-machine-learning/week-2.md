---
title: Week 2 - Vectors are objects that move around space
date: 2021-09-12 00:00
category: reference/moocs
status: draft
parent: linear-algebra-machine-learning
modified: 2023-04-09 00:00
---

## Intro

* The module covers the following topics:
  * Calculating the Magnitude (aka Vector Modulus).
  * Combining vectors using the [Dot Product](../../../../permanent/dot-product.md).
  * [Vector Scaling](../../../../permanent/vector-scaling.md) and Vector Projections
  * Looking at the [Vector](../../../../permanent/vector.md) set we use to define space called the [Basis Vectors](../../../../permanent/basis-vectors.md).
  * Understanding Linear Independence and Linear Combinations.

## Finding the size of a vector, its angle, and projection

### Modulus and inner product

* A vector has two properties: its length and its direction.
* [Vector Magnitude](../../../../permanent/vector-magnitude.md) (0:35-2:54)
    * A vector can be described in terms of its unit vectors. In this example, they're $\hat{i}$ and $\hat{j}$:
        $r = a\hat{i} + b\hat{i}$
    * From [Pythagoras Theorem](permanent/pythagoras-theorem.md), we can calculate the magnitude (size) of the vector as $\|r\| = \sqrt{a^2 + b^2}$
    * Vectors are mostly written as column vectors: $\begin{bmatrix} a \\ b \end{bmatrix}$
    * Doesn't matter if our vector contains dimensions in space or things with different physicals units like time and price: we define the size of a vector through the square root of the sums of the squares of its components.
* [Dot Product](../../../../permanent/dot-product.md) (02:54-10:00)
    * If you have 2 vectors $r = \begin{bmatrix}r_i \\ r_j\end{bmatrix}$ and $s = \begin{bmatrix}s_i \\ s_j\end{bmatrix}$, the dot products are defined as: $r \cdot s = r_is_i + r_js_j$
    * The dot product has the following properties:
        1. It's commutative: $r \cdot s = s \cdot r$
        2. It's distributive over additional. If you had a 3rd vector $t$, then $r \cdot (s + t) = r \cdot s + r \cdot t$
        3. It's associative over scalar multiplication
            1. $r \cdot (as) = a(rs)$
    * A vector dot product against itself is equal to the magnitude of a vector squared. By taking the square root, you have the magnitude of the vector.

### Cosign and dot product

* Cosign rule from algebra ([Law Of Cosines](../../../../permanent/law-of-cosines.md)):
    * If you have a triangle with sides a, b and c and angle between a and b $\theta$ then: $c^2 = a^2 + b^2 - 2ab \cos \theta$

        ![Cosine rule example](/_media/laml-cosine-rule.png)

* In the triangle above, we can use [Vector Subtraction](../../../../permanent/vector-subtraction.md)
* So $c^2  = |r-s|^2$, which means:
    * $|r-s|^2 = |r|^2 + |s|^2 - 2|r| |s| \cos\theta$
    * We can replace $|r-s|^2$ with the dot product of itself $(r-s)(r-s)$
    * We can then multiply it out: $r \cdot r - r \cdot s - s \cdot r - s \cdot -s$
    * Which can be simplified to: $|r|^2 - 2s.r + |s|^2$
    * We can compare to right hand side: $|r|^2 - 2s.r + |s|^2 = |r|^2 + |s|^2 - 2|r| |s| \cos\theta$
        * Which can be simplified to: $-2s.r = -2|r| |s| \cos\theta$
        * Or $r.s = |r| |s| \cos \theta$
* If $r$ and $s$ were unit vectors, then $r.s = \cos \theta$!
* If $\theta=90°$, we know that $\cos 90°=0$, so the dot product of 2 orthogonal vectors is 0!
* Since we know $\cos 0° = 1$, the dot product of 2 unit vectors pointing in the same direction is 1.
* We know $\cos 180 ° = -1$ , so the dot product of 2 opposite vectors is -1.

### Projection

* [Scalar Projection](permanent/scalar-projection.md) (00:00-04:04)
    * Scalar projection is the amount one vector "goes along" another.
        * Draw a line straight down from the tip of vector $s$ onto vector $\vec{r}$. How far along $\vec{r}$ does it land?

            ![Scalar projection example](/_media/laml-scalar-projection.png)

        * Using Pythagoras Theorem, we know that $\theta = \frac{adj}{hyp}$
            * In the above image, the hypotenuse is the length of s: $|\vec{s}|$
               so $\cos\theta = \frac{adj}{|\vec{s}|}$
               which we can rearrange to $\text{adj} = |\vec{s}|\cos\theta$

    * We note that the dot product is $|r| * \text{adj}$:
         * $r.s = |r| \underbrace{|s| \cos\theta}_{\text{adj}}$
    * So the dot product is: "the projection of s onto r" * "size of r": $|r| \times \text{projection}$
    * You can rearrange the expression to: $\frac{r.s}{|r|} = |s|\cos\theta$, to get the scalar projection.
    * That explains why the [Dot Product](../../../../permanent/dot-product.md) is also called the "Projection Product".
* [vector-projection](permanent/vector-projection.md) (04:04-05:52)
    * Allows you to include the direction of $r$ into the scalar projection.
    * Defined to be $\vec{r} \frac{\vec{r} \cdot \vec{s}}{|r||r|}$
    * Or $\vec{r} \frac{\vec{r} \cdot \vec{s}}{\vec{r} \cdot \vec{r}}$

## Changing the reference frame

### Changing basis

* [Coordinate System](Coordinate System) (00:00 - 03:35)
    * Coordinate system is defined by 2 vectors called the [Basis Vectors](../../../../permanent/basis-vectors.md). In the image, they're $\hat{e}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\hat{e}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

        ![Basic vectors](/_media/laml-basic-vectors.png)

    * If our space had more dimensions, you could have more basis vectors.
    * We can think of $\vec{r}$ as being some amount of the basis vectors: $r_e = 3\hat{e}_1 + 4\hat{e}_2  = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$
    * Choice of $\hat{e}_1$ and $\hat{e}_2$ is arbitrary. It could be at 90° or a different length. We would still define R as the sum of vectors used to define space.
    * If we defined another set of basis vectors $b_1$ and $b_2$, we could describe $\vec{r}$ in terms of those vectors. The numbers in $\vec{r}$ would be different.
        $r_b = \begin{bmatrix} ? \\ ? \end{bmatrix}$
    * We can note that the vector r exists in a different space with another set of numbers.
* [Changing Basis](../../../../permanent/changing-basis.md) (03:36-10:32)
    * We can use the dot product to find the numbers for a [Vector](../../../../permanent/vector.md) in a different space if you know the new basis vectors in terms of the original.
        * In this example, $b_2 = \begin{bmatrix} -2 \\ 4 \end{bmatrix}$ (or `-2, 4` $e_2$s) and $b_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ (or `2, 1` $e_1$s)

            ![New basic vectors to calculate r for](/_media/laml-new-basic-vectors.png)

    * The new set of basis vectors $b$ must be 90° to each other. We can use the dot product to find the new numbers for $r$ in a new basis $b$ if we know what $b$ is in terms of $e$.
        * If they aren't 90 degrees to each other, we have to use matrices and a transformation of the axis.
    * If we first project $\vec{r}$ onto $b_1$, we get a scalar projection that describes how much of $b_1$ you need. Vector projection gives you a projection in $b_1$ of that length.
    * Same with $b_2$
    * Maths works like this:
        * Project $r_e$ onto $b_1$, which gives you a scalar projection. That is ${r_b}_{1}$
            * Then vector projection gives you vector in the direction of b_1, which is the length of r.
        * Do that for all ${r_e}_{n}$ to get the vector $r_b$
        * For example:
            * Get the scalar projection of $r_e$ onto $b_1$ $\frac{r_e \cdot b_1}{b_1^2} = \frac{3 \times 2 + 4 \times 1}{2^2 + 1^2} = \frac{10}{5} = 2$
                * Note you get the vector projection of $\frac{r_e \cdot b_1}{{b_1}^2} = 2 \begin{bmatrix}2 \\ 1\end{bmatrix} = \begin{bmatrix}4 \\ 2 \end{bmatrix}$
            * Then $\frac{r_e \cdot b_2}{|b_2|^2} = \frac{10}{2} = \frac{1}{2}$
            * So $\vec{r_e}$ in the basis $b_1$ and $b_2$ is $\begin{bmatrix} 2 \\ 0.5 \end{bmatrix}$

### Basic, vector space, and linear independence

* A basis is a set of n-vectors that:
    1. Are not linear combinations (they're linearly independent)
    2. Span the space: the space is then n-dimensional
* Linear Independence (0:42-2:37)
    * A vector is independent of another if you cannot get the 2nd vector by taking multiplies of the first.
        * $b3 \neq a_1b_1 + a_2b_2$ for any $a_1$ or $a_2$
        * In other words, $b_3$ does not lie on the plane spanned by $b_1$ and $b_2$
* Basis vectors must be linearly independent, but it doesn't mean that they have to be unit vectors of length one or orthogonal, but life is much easier if they are.

### Applications of changing basis

* If you have some 2d data points that lie on a straight line, you can map how far each data point maps along a line.
    * Distance from the line is considered noise.
    * Noise tells you how good the line fit is.
* Goal of a neural network would be to extract the most important features as basis vectors.

## Summary

* We've learned about vectors being objects that describe where we are in space.
* We've defined vector attributes like additional and scaling a vector by a number.
* Found the magnitude of Modulus of a vector.
* Done the dot scalar and vector projection product.
* Defined the basis of vector space and its dimension
* Ideas of linear independence and linear combinations.
* Used projections to look at one case of changes from one basis to another (where basis vectors are orthogonal).
