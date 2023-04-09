---
title: Attachment (Roblox Docs)
date: 2021-11-25 00:00
category: reference/articles
status: draft
---

These are notes from [Attachment](https://developer.roblox.com/en-us/api-reference/class/Attachment) from the Roblox Developer Hub.

Attachment defines a point and orientation relative to a parent BasePart.

Roblox stores the offset in the `CFrame` property.

```
local attachment = Instance.new('Attachment')
attachment.Parent = game.Workspace.Part
print(attachment.CFrame)
```

We can set the offset through other properties like `WorldCFrame`.

Attachments are used by some [Roblox Constraints](../../permanent/roblox-constraint.md) and are valid parents for some objects:

* `ParticleEmitter` and `Fire` if you want to emit from an attachment instead of a part's [Roblox CFrame](../../permanent/roblox-cframe.md).
* `PointLight` and `SpotLight` to shine from attachment.
* `Sound` to change the focal point of sound.

Key properties:

* `Axis` - represents the direction of X-Axis relative to Attachment's `Attachment.Rotation`, as a unit Vector3 with a length of 1.
* `CFrame` - CFrame offset of the attachment.
* `Orientation` - represents orientation relative to parent.
* `Position` - positional offset, relative to parent.
* `SecondaryAxis` - represents direction of Y-Axis relative to `Rotation` as unit [Roblox Vector3](../../permanent/roblox-vector3.md).
* `Visible` - toggle visibility of the Attachment.
