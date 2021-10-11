---
title: Unit Vector
---

Unit vector is a vector with a length of 1. Denoted by a lowercase letter with a "hat" like: $\hat{\mathbf{v}}$

When you normalise a non-zero vector, it rusn a unit vector in the vector's direction.

$\hat{u} = \frac{u}{\|u\|}$

In a [[Roblox Vector3]], a `Unit` method returns the normalised vector. In other words, a vector that only has the direction.

```lua
> print(Vector3.new(0, 1000, 0).Unit)
0, 1, 0
```