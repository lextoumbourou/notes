---
title: Scalar Projection
date: 2026-09-26 09:05
modified: 2026-09-26 09:05
status: draft
---

**Scalar Projection** is the amount one vector "goes along" another: the length of the shadow that vector $\vec{s}$ casts onto vector $\vec{r}$.

It's calculated using the [Dot Product](dot-product.md) divided by the [Vector Magnitude](vector-magnitude.md) of the vector being projected onto:

$$
\text{scalar projection of } \vec{s} \text{ onto } \vec{r} = \frac{\vec{r} \cdot \vec{s}}{|\vec{r}|} = |\vec{s}| \cos\theta
$$

where $\theta$ is the angle between the two vectors. The result is a single number (a [Scalar](scalar.md)), and it's negative when the angle is more than 90°.

See [Week 2 - Vectors are objects that move around space](../reference/moocs/coursera/linear-algebra-machine-learning/week-2.md).
