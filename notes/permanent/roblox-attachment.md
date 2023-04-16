---
title: Roblox Attachment
date: 2021-12-12 00:00
summary: An object that describes a point and orientation relative to a BasePart
tags:
  - Roblox
cover: /_media/roblox-attachment-cover.png
hide_cover_in_article: true
---

An `Attachment` in Roblox is an object that describes a point and orientation in space relative BasePart's [Roblox CFrame](roblox-cframe.md).

Consider a `Fire` that you want to burn from the back of a `Part`. If I attach a `Fire` instance directly to a `Part`, it will always burn from the `Part`'s center of the Part. Instead, I can create an `Attachment` parented to the `Part` and position it -2 studs along the X-axis and one stud along the Y-axis from the Part's center.

The `Attachment` remains in the same place relative to the `Part` as the part moves.

<video controls loop autoplay><source src="/_media/roblox-attachment-low.mp4" type="video/mp4"></video>

Here's how it looks in code:

```lua
--!strict

-- Create Part.
local part: Part = Instance.new('Part')
part.Anchored = true

-- Create an Attachment behind the Part.
local attachment: Attachment = Instance.new('Attachment')
attachment.Position = Vector3.new(-2, 1, 0)
attachment.Parent = part

-- Attach fire to Attachment instead of Part.
local fire: Fire = Instance.new('Fire')
fire.Size = 1
fire.Heat = 1
fire.Parent = attachment

part.Parent = game.Workspace

-- Code to move Part around below.
```

A new attachment can be created using `Instance.new('Attachment')`.

An attachment's position can be updated using the `Position` attribute or editing the `CFrame` attribute.

There are many items that you can parent directly to an `Attachment` instead of a `Part`:

* Any `ParticleEmitters` can be parented directly to an `Attachment.`
* `Sound` objects allow audio to play directly from the `Attachment`'s location.
* `PointLight` and `SpotLight` allow light to shine from a specific point on a Part.

[Roblox Constraint](Roblox Constraint) objects rely on Attachments.

The [Roblox Accessory](Roblox Accessory) system utilizes attachments to position accessories on a character's body parts.
