---
title: Understanding CFrames
date: 2021-08-10 00:00
status: draft
tags:
  - Roblox
  - LinearAlgebra
---

Notes from [Understanding CFrames](https://developer.roblox.com/en-us/articles/Understanding-CFrame)

* A CFrame, short for Coordinate Frame is what Roblox uses for rotating and positioning 3D objects.
* CFrame contains global X, Y and Z coordinates + rotation data for each axis.
* CFrames have helpful functions for working with objects in the 3D space.
* Examples:
    * Aiming projectiles
    * Moving camera to focus on specific NPCs
    * Placing status indicator above a player's head to show paralyzed, boosted, poisoned.
* CFrame Basics
    * `CFrame.new()` creates object at position 0, 0, 0.
    * You can also pass X, Y and Z are postional arguments:
        * `CFrame.new(0, 20, 0)` - move 20 studs in the air.
        * Note that in Roblox, when it displays markers that visualise position in 3d place, like the ones in the Move tool, X is Red, Y is Green and Z is Blue.
    * Or pass in a `Vector3.new(X, Y, Z)` instead. That is required when using the "look at" style described below.
    * `CFrame.Angles()` to provide a rotational angle in radians for desired axes.
        * Rotating parts works by rotating around an axis.
        * You must pass radians instead of degrees (use `math.rad(degrees)` to convert).
    * Facing towards a point
        * Can create a part then point its front surface by passing a second Vector3 argument (`Position` is a shortcut for the Position Vector3):

            ```
            game.Workspace.Block.CFrame = CFrame.new(
                Vector3.new(0, 10, 0),
                game.Players.lexandstuff.Character.HumanoidRootPart.Position)
            ```

    * Can also offset an object from position of another object:
        `CFrame.new(humanoidRootPart.Position) + Vector3.new(0, 2, 0)`
* Dynamic CFrame Orientation
    * `CFrame.new()` and `CFrame.Angles()` can only position/rotate an object at a specific orientation
        * CFrame functions can allow you to do things in relation to other CFrame positions
    * `CFrame:ToWorldSpace()`
        * Transforms an object's CFrame to a new world orientation.
        * Useful for offsetting part relative to another object.
        * Can also be used for rotating an object relative to itself.
    * `CFrame.lerp`
        * Position a CFrame between 2 points using a linear interpolation (lerp).
        * `redBlock.CFrame = greenCube.CFrame:Lerp(cyanCube.CFrame, 0.7)``
