---
title: CFrame Math Operations
date: 2021-08-21 00:00
status: draft
tags:
  - Roblox
  - LinearAlgebra
---

Notes from [CFrame Math Operations](https://developer.roblox.com/en-us/articles/CFrame-Math-Operations).

## Components of a CFrame

* CFrame made up of 12 separate numbers - "components"
    * Can get components of CFrame with `cf:components()`
    * Can also pass all 12 args in to `CFrame.new` constructor
* First 3 numbers are position.
* Next 3 is `cf.rightVector`
* Next 3 after that is `cf.upVector`
* Next 3 after that is `cf.lookVector`
* CFrame * CFrame
    * CFrames are 4x4 matrices of this form:

    ```
    [
      [m11, m12, m13, x],
      [m21, m22, m23, y],
      [m31, m32, m33, z],
      [0, 0, 0, 1]
    ]
    ```

    * 2 CFrames can be multiplicated with [Matrix Multiplication](../../permanent/matrix-multiplication.md)
* CFrame * Vector3
    * Results in Vector3.
    * The first value multiplies and adds the rightVector, 2nd upVector and 3rd rightVector
* CFrame + or - Vector3
    * Adds to x, y, z and leaves the rotational part unchanged.
* Inverse of a CFrame
    * Matrices are generally not commutative (with the exception of identity and some matrices)
        * If you multiply a matrix by its inverse, either pre or post, it was always return the identity CFrame

        ```
        local cf = CFrame.new(1, 2, 3) * CFrame.Angles(math.pi/2, 0, 0)
        
        print(cf*cf:inverse())

        print(cf:inverse()*cf)
        ```

* Reverting to original values
    * If you had, `local cf = cf1 * cf2`, and were only given `cf` and `cf1`, could you find `cf2`?
        * `cf1:inverse() * cf = cf1:inverse() * cf1 * cf2`
        * `cf1:inverse() * cf = CFrame.new() * cf2` since `cf:inverse() * cf = identityCFrame`
        * `cf1:inverse() * cf = cf2` since identityCFrame * cf = cf`
* Rotating a door
    * If you used `CFrame.Angles` on a part, it would spin from the center.

    ```
    local door = game.Workspace.Door

    game:GetService("RunService").Heartbeat:connect(function(dt)
        door.CFrame = door.CFrame * CFrame.Angles(0, math.rad(1)*dt*60, 0)
    end)
    ```

    * But we can rotate a hinge. Then to calculate the offset of the door from the un-rotated hinge

    ```
    local door = game.Workspace.Door
    local hinge = game.Workspace.Hinge

    local offset = hinge.CFrame:inverse() * door.CFrame; -- offset before rotation
    game:GetService("RunService").Heartbeat:connect(function(dt)
        hinge.CFrame = hinge.CFrame * CFrame.Angles(0, math.rad(1)*dt*60, 0) -- rotate the hinge
        door.CFrame = hinge.CFrame * offset -- apply offset to rotated hinge
    end)
    ```

* CFrame methods
    * `CFrame:ToObjectSpace()`
        * Equivalent to `CFrame:inverse() * cf`
        * Calculates the offset to get from CFrame to cf
    * `CFrame:ToWorldSpace()`
        * Equivalent of `CFrame * cf`
    * `CFrame:PointToObjectSpace()`
        * Equivalent of `CFrame:inverse() * v3`
    * `CFrame:PointToObjectSpace()`
        * Takes a point in 3D space and makes it relative to `CFrame.p`
        * Equivalent to `(CFrame - CFrame.p):inverse() * (v3 - CFrame.p)`
    * `CFrame:PointToWorldSpace()`
        * Equivalent to `CFrame * v3`
        * Apply an offset without rotational aspect.
    * `CFrame:VectorToObjectSpace()`
        * Equivalent to `(CFrame - CFrame.p):inverse() * v3`
        * Does not make `v3` relative to `CFrame.p`. Assumes input `v3` already relative.
    * `CFrame:VectorToWorldSpace()`
        * Equivalent to `(CFrame - CFrame.p) * v3*`
