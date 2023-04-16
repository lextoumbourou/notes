---
title: Raycasting
date: 2020-11-05 00:00
tags:
  - Computer Graphics 
---

Raycasting is a rendering technique that casts a series of [Ray](ray.md)s for each vertical slice of the screen in the direction a player is facing. Any item that is "hit" by the ray is then rendered onto the player's screen with a size dependant on the distance to from the player. It is a computationally inexpensive method of 3d rendering that was popularised by Wolfenstein 3d.

In the Roblox Engine, a similar concept is given the name raycasting ([Roblox Raycasting](roblox-raycasting.md)), which cast rays from a position in 3d space in a given direction and returns the first Part or Terrain that is hit by the ray. It's commonly used to build guns and other projectile weapons, as well as detect terrain changes beneath a player's feet.

References:

* [Lode's Computer Graphics Tutorial](../reference/articles/Lode's Computer Graphics Tutorial.md) (Raycasting)
