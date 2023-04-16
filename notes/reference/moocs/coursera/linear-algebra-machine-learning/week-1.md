---
title: Introduction to Linear Algebra and to Mathematics for Machine Learning
date: 2018-04-10 00:00
category: reference/moocs
status: draft
parent: linear-algebra-machine-learning
---

# Introduction to Linear Algebra and Mathematics for Machine Learning

## Introduction

* Professor: David Dye
* Lots of data in the world: need ways to make sense of it.
* Linear algebra (vector/matrix algebra) important in ML.
* Multivariate calculus is required to understand how something you're optimizes changes with respect to variables.
* Most DS and ML courses have these as a prerequisite: this course fills the gap.

## Motivations for Linear Algebra

* Price discovery by solving simultaneous equations:

    $$
  \begin{aligned}
  a = \text{price of apple} \\
  b = \text{price of banana} \\
  2a + 3b = 8 \\
  10a + 1b = 13
  \end{aligned}
  $$

* Example of linear algebra problem: have constant linear coefficients 2, 3, 10, 1, that relate the input variables $a$ and $b$ to the output values 8 and 13.
* If you had a vector that describes the price of apples and bananas:

    $\begin{bmatrix} a \\ b \end{bmatrix}$

* Can write the equation as a matrix problem:

    $$
    \begin{pmatrix}
    2, 3\\
    10, 1
    \end{pmatrix}
    \begin{bmatrix}
    a\\
    b
    \end{bmatrix}
    = 
    \begin{bmatrix}
    8\\
    13\end{bmatrix}
    $$

* In this course, we'll learn how to solve this problem in the general case.
* Fitting an equation to some data:
    * Useful for describing a population without requiring data.

## Getting a handle on vectors

* If you have a normal distribution with mean $\mu$ and standard deviation $\sigma$ of a population, the function that plots the curve is as follows:

    $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp({ \frac{-(x-\mu)}{2\sigma^2})}$$

* How do you find a function that finds the optimal $\mu$ and $\sigma$?
  * Determine some function that tells you how far off you are: eg sum of squared differences.
  * Then, use calculus to walk toward the most optimal solution.
* Vectors don't just to describe objects in geometric space, they can describe directions along any sort of axis
    * Can think of them as just lists.
    * Space of all possible cars: `[cost_in_euros, emissions, top_speed, ...]`
        * Computer science view of vectors. See [Vector](../../../../permanent/vector.md).
    * Spatial view is more familiar for physics.
    * Einstein conceived of time being another dimension. Space-time is a 4-dimensional vector.

## Operations with vectors

* Vector can be thought of as an object that moves about space.
  * Space = physical space or data space.
  * Example vector might include properties of a house: 120 sqm^2, 2 bedrooms, one bathroom, $150k: `[120, 2, 1, 150]`
* Vector should obey two rules:
      1. We can add vectors (see [Vector Addition](../../../../permanent/vector-addition.md)).
          * associative: doesn't matter what order you add
              `vector_1 + vector_2 = vector_2 + vector_1`
          * Since subtraction is just the addition of the negative, i.e.` r - r = r + (-r)`, the same rules apply to subtraction as addition.
      2. We can multiply vectors by a scalar (see [Vector Scaling](../../../../permanent/vector-scaling.md)).
          * Multiples each value in the vector by some scalar: `2 * [1, 2] = [1 * 2, 2 * 2] = [2, 4]`
