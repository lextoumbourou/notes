---
title: Abstract Vector Spaces
date: 2022-01-24 00:00
status: draft
---

Revisiting question: what are vectors?
* Is it 2d arrow on a flat plane, which we can describe with coordinates?
* Or, is it fundementally a pair of real numbers which can visualed as an arrow on a flat plane?

On the one hand, the list of numbers makes things unambiguous. It makes it easier to work with larger dimensions compared to requiring visualisations.

On the other hand, for people who're actually working with linear algebra, you are often dealing with spaces independant from the coordinate system. The coordinates are arbitrary and dependant on what you choose as [Basis Vectors](../../../permanent/basis-vectors.md).

Core topics in linear algebra are indifferent to the choice of basis. The determinate is how much a transofmration scales areas, Eigenvectors are the ones that stay on their span during a transformation. You can freely change the coordinate system without changing the underlying values of each.

If vectors aren't list of real numbers, and their underlying essence is more spatial. What do mathematians mean when they use the word "space"?

Firstly, let's look at functions.

In one sense, functions are another type of vector. Like how you can add 2 vectors together, you can also add 2 functions together.

Can we also apply things like linear transformations, null space, dot products, eigen-things to functions?

In Calculsu, the derivative takes one function and returns another (functions also known as operators).

What does it mean for a transformation of functions to be linear?

Additivity: $L(\vec{v} + \vec{w}) = L(\vec{v}) + L(\vec{w})$
    * If you add 2 vectors, then apply transform to sum, it's the same as adding the transform of $\vec{v}$ and $\vec{w}$
Scaling: $L(c\vec{v}) = cL(\vec{v})$
    * If you scale a vector $\vec{v}$ then apply transform, it's the same as scaling transformed $\vec{v}$

You hear it described as "preserving" addition and scalar multiplication.

The idea of grid lines remaining parallel and evenly spaced is really to just illustration of what these 2 properties mean with points in 2d space.

The important consequence of property is that linear transformation is completely described by where it takes the basis vectors.

This is as true for functions as it is for arrows.

For example, calculus students know that derivative is additive and has the scaling property, even if they haven't heard it phrased that way.

If you add 2 functions, then first take the derivative, it's the same as taking the derivative of each separately, then adding the results.

Let's see what it looks like to describe a derivative with a matrix
