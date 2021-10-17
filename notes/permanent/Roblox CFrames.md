---
title: Roblox CFrames
date: 2021-10-16 12:00
tags:
  - Roblox
  - LinearAlgebraApplications
status: draft
---

In Roblox, a CFrame is an object that represents the position and rotation of a [[Roblox Part]].

A new CFrame can be created with the `new` constructor.

At it simplest, it's a replacement for setting position `Position`:

```
part.Position = Vector3.new(0, 0, 0)
```

```
part.CFrame = CFrame.new(0, 0, 0)
```

Or, the `Angles` constructor generates a rotated CFrame sitting at point `0, 0, 0`:

```
part.CFrame = CFrame.Angles(0, math.rad(90), 0)
```

Note that the rotation must be expressed in [[Radians]].

You can combine a CFrame that specifies position with a CFrame that specifies Angle, using the `*` operator:

```
--- Place a part at 0, 10, 0 with Orientation 0, 90, 0.
part.CFrame = CFrame.new(0, 10, 0) * CFrame.Angles(0, math.rad(90), 0)
```

Later in the article, we'll look at what is happening under the hood when multiplying 2 CFrames.

You can create a CFrame that looks at another part using the `lookAt` constructor:

```
local positionToLookAt = otherPart.Position

part.CFrame = CFrame.lookAt(Vector3.new(0, 10, 0), positionToLookAt)
```

A CFrame can be used to calculate relative position from CFrame. For example, if I want to put a part in front of a player's face, no matter which direction they're facing (even if they're upside down), I can use the `ToWorldSpace()`:

```
local relativePositionOfNewPart = CFrame.new(0, 0, 10)
part.CFrame = character.Head.CFrame:ToWorldSpace(relativePositionOfNewPart)
```

A CFrame is represented as a matrix under the hood, with 4 columns:

1. Right vector (rX, rY, rZ)
2. Up vector (uX, uY, uZ)
3. Look (front) vector (lX, lY, lZ)
4. Position vector (x, y, z)

Plus an extra row that allows for convenient [[Matrix Multiplication]]:

$\begin{bmatrix}rX & uX & lX & x \\ rY & uY & lY & y \\ rZ & uZ & lZ & z \\ 0 & 0 & 0 & 1\end{bmatrix}$

The right, up and look vectors are orthonormal (perpendicular) [[Basis Vectors]] that make up the object space for the CFrame.

You can see why multiplying a position CFrame by a Angle CFrame returns a rotated CFame in this example:

(to do)

All the components of a CFrame can be returned using the `CFrame:Components()` method.

It returns the components of each vector in this order:

```
local x, y, z, rightX, rightY, rightZ, upX, upY, upZ, lookX, lookY, lookZ = cf:Components()
```

Note that this can be used for serialising CFrames (ie for storing a CFrame position in a datastore).

```
local x, y, z, rightX, rightY, rightZ, upX, upY, upZ, lookX, lookY, lookZ = cf:Components()
CFrame.new(x, y, z, rightX, rightY, rightZ, upX, upY, upZ, lookX, lookY, lookZ)
```




