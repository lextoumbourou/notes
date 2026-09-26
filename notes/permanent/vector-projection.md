---
title: Vector Projection
date: 2026-09-26 08:50
modified: 2026-09-26 08:50
status: draft
---

The **Vector Projection** of a [Vector](vector.md) $\mathbf{a}$ onto another vector $\mathbf{b}$ is the component of $\mathbf{a}$ that points in the direction of $\mathbf{b}$. You can think of it as the "shadow" $\mathbf{a}$ casts on the line through $\mathbf{b}$.

It's calculated using the [Dot Product](dot-product.md):

$$
\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} \mathbf{b}
$$

The scalar projection, $\frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{b}|}$, gives just the (signed) length of that shadow, where $|\mathbf{b}|$ is the [Vector Magnitude](vector-magnitude.md) of $\mathbf{b}$.
