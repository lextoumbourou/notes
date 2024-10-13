---
title: Convex Hull
date: 2024-01-26 00:00
modified: 2024-01-26 00:00
summary: the smallest polygon that contains a set of points
tags:
  - ComputerScience
  - Algorithms
cover: /_media/convex-hull.png
hide_cover_in_article: true
---

The **Convex Hull** is the smallest polygon that contains a set of points.

Think of some nails poking out of a board. If you place a rubber band around all nails, the polygon shape the rubber band makes is the convex hull.

![](../_media/convex-hull.png)

One real-world application in game programming is **Collision Detection**. Since computing collisions can be expensive, a convex hull is calculated around the 3d model of an object to provide an approximate collision region, reducing the number of computations required.

This example from the Roblox docs shows the collision region for a 3d model with the `Hull` collision option.

| Original Mesh                       | **Hull Collision Region**                                        |
| ----------------------------------- | --------------------------------------- |
| ![A 3d mesh](../_media/3d-model-no-hull.png) | ![Convex Hull collision](../_media/3d-model-convex-hull.png) |

*This example of a 3d model to its hull collision model comes from the [Roblox docs](https://create.roblox.com/docs/workspace/collisions#mesh-and-solid-model-collisions).*

---

## Recommended Reading

[Introduction to Algorithms, Third Edition](https://amzn.to/3HyDauB)

![Intro to Algorithms cover](../_media/intro-to-algorithms-3rd.png)

**Chapter 1** has an introduction to Convex Hull. **Chapter 33** details some algorithms for computing Convex Hulls
