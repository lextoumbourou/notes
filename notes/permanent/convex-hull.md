---
title: Convex Hull
date: 2024-01-26 00:00
modified: 2024-01-26 00:00
summary: the smallest polygon that contains a set of points
tags:
  - Algorithms
cover: /_media/convex-hull.png
hide_cover_in_article: true
---

The **Convex Hull** is the smallest polygon that contains a set of points.

Think of some nails on a board. If you were to place a rubber band around them all, the **Convex Hull** would be the polygon shape the rubber band makes.

![](../_media/convex-hull.png)

One real-world application in game programming is [[Collision Detection]]. Since computing collisions can be expensive, a convex hull is calculated around the 3d model of an object to provide an approximate box for detecting character collisions, reducing the number of computations required.

This example from the Roblox docs shows the collision region for a 3d model with the `Hull` collision option.

| Original Mesh                       | **Hull**                                        | 
| ----------------------------------- | --------------------------------------- |
| ![A 3d mesh](../_media/3d-model-no-hull.png) | ![Convex Hull collision](../_media/3d-model-convex-hull.png) |
*This example of a 3d model to its hull collision model comes from the [Roblox docs](https://create.roblox.com/docs/workspace/collisions#mesh-and-solid-model-collisions).*


## References

For more about Convex Hulls and algorithms for computing them, see Chapter 33 of [Introduction to Algorithms, Third Edition](https://amzn.to/3HyDauB).

![Intro to Algorithms cover](../_media/intro-to-algorithms-3rd.png)
