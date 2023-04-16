---
title: Function
date: 2023-04-14 00:00
tags:
  - DiscreteMath
summary: A rule that matches inputs with outputs.
modified: 2023-04-14 00:00
cover: /_media/function-cover.png
---

In Math, a function is a rule that associates inputs with outputs.

For example, the function $f$ inputs a number, $x$, and outputs its square:

$$
f(x) = x^2
$$

Functions are a "well-behaved relation", which means that for each input, there is exactly one output.

We can express functions as a mapping from one [Set](set.md) $A$ to another set $B$.

$$f : A \rightarrow B$$

In the square function from earlier, the function maps from the set of real numbers to the set of non-negative real numbers number (as $-x^{2}$ is a non-negative number).

$f : \mathbb{R} \rightarrow \mathbb{R}_{\geq 0}$

Note that $\mathbb{R}$ is one of the [Special Infinite Sets](special-infinite-sets.md).

* The [Domain of a Function](function-domain.md) is the input set that the function maps from, expressed as $D_f = A$.
* the [Co-Domain of Function](function-codomain.md) is the output set of a function, expressed as $co-Df = \mathbb{R}_{\geq 0}$.
* The range of a function, $R_f$, is the set of all possible outputs.

Functions are a fundamental building block of most programming languages.

In some languages, we can express functions by their input and output types.

For example, this is the Python-typed equivalent of the function from earlier:

```python
def f(y: int) -> int:
    return y ** 2
```

## Plotting Functions

We can visualise functions by plotting the inputs on the x-axis and the outputs on the y-axis.

## Common Functions

* [Linear Function](linear-function.md).
* [Quadratric Functions](quadatric-functions.md)
* [Exponential Function](exponential-function.md)

## One-to-one

We consider a function "one-to-one" or "injective" if there is exactly one output for every element in the input space and no two different inputs have the same output.

## Onto

We consider a function "onto" or "surjective" if every element of the [Co-Domain of Function](function-codomain.md) has a matching element of the input space or [Domain of a Function](function-domain.md).

## Bijective

We call a function Bijective if it is both One-to-one and Onto.
