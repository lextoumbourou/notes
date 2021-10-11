---
title: CFrame (Roblox)
date: 2021-08-21
status: draft
tags:
  - GameMath
  - LinearAlgebra
---

A CFrame in Roblox is used to describe a Part location and rotation of a part in 3d space.

The CFrame of any part can be get or set  using the `CFrame` attribute.

I can place a part at the origin (a point with coordinates `(0, 0, 0)`) using CFrame:

```
game.Workspace.Part.CFrame = CFrame.new(0, 0, 0)
```

That's equivalent to setting the position:

```
game.Workspace.Part.Position = Vector3.new(0, 0, 0)
```

Note that in Roblox the position of an part refers to a point in the middle of the part.

The CFrame `new` constructor also accepts a 2nd argument which specifices the `lookVector`, which is a Vector3 coordinate that specifies a point that the part will be facing.

A common use case of this is to set the Camera CFrame to face a direction. 

```
game.Workspace.Camera.CFrame = CFrame.new(game.Workspace.CameraLocation.Position, LocalPlayer.Character.HumanoidRootPart.Position)
```

A CFrame that simply defines the rotational position can be created with `CFrame.Angles`.

For example, to rotation a part around the Y-Axis by 90 degrees, we can do the following:

```
game.Workspace.Part = CFrame.Angles(0, math.rad(90), 0)
```

Note that the rotation is specified in Radians. The `math.rad` function converts from degrees to radians.

The CFrame is a matrix that describes a position as a [[Roblox Vector3]]

It also contains a right-vector. A [[Unit Vector]] pointing in the Right-direction.

A up Vector, which is a [[Unit Vector]] pointing up.

Or a lookVector, a vector facing back.

```
local cf = CFrame.new(0, 0, 0)
local x, y, z, m11, m12, m13, m21, m22, m23, m31, m32, m33 = cf:components()

local right = Vector3.new(m11, m21, m31) -- This is the same as cf.rightVector
local up = Vector3.new(m12, m22, m32) -- This is the same as cf.upVector
local back = Vector3.new(m13, m23, m33) -- This is the same as -cf.lookVector
```

