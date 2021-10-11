---
title: Roblox Dot Product
date: 2021-08-30 23:30
tags:
  - Roblox
  - LinearAlgebra
---

In Roblox, we can calculate the [[Dot Product]] of 2 Vector3 instances ([[Roblox Vector3]]) using the `Dot` method on either instance. Since [[Vector]]s are commutative, the order doesn't matter. 

Since we know that Dot Products equals the product of the magnitude of both vectors * cos(angle between vectors), we can use the dot product to determine the angle between vectors. For example:

```lua
local angleBetweenVectors = math.acos(A:Dot(B) / (A.Magnitude * B.Magnitude))
```

If A and B were [[Unit Vector]]s, we wouldn't need to worry about dividing the magnitudes since we know they'd be equal to 1.

```lua
local angleBetweenVectors = math.acos(A.Unit:Dot(B.Unit)
```

A use case for this is to detect when a character is turning.

In this example, we get the Humanoid's MoveDirection vector. That's a unit vector along the X/Z axis, which describes the direction.

```lua
local previousMoveDirection = Humanoid.MoveDirection
```

 Then, ten frames later, we get the new MoveDirection vector and do a dot product against the first.
 
```lua
local currentMoveDirection = Humanoid.MoveDirection
  
local dotProduct = currentMoveDirection:Dot(previousMoveDirection)
```

If the angle between vectors is greater than 0 degrees, we're turning!

```lua
local angleOfTurn = math.acos(dotProduct)
if isMoving and angleOfTurn > 0 then
    -- Do something with this information
end
```

Here's a complete example of a LocalScript in StarterCharacterScripts:

```lua

local RunService = game:GetService("RunService")
local Humanoid = game.Players.LocalPlayer.Character:WaitForChild('Humanoid')

function watchForPlayerTurning()
	local frameCount = 0

	local prevDirection = Humanoid.MoveDirection

	return RunService.RenderStepped:Connect(function()
		if frameCount % 10 == 0 then
			local currentDirection = Humanoid.MoveDirection
			local isMoving = currentDirection.Magnitude > 0 and prevDirection.Magnitude > 0

			local angleOfTurn = math.acos(currentDirection:Dot(prevDirection))

			if isMoving and angleOfTurn > 0 then
				print("Character is turning at "..math.deg(angleOfTurn).." degrees")
			end

			prevDirection = Humanoid.MoveDirection
		end

		frameCount = frameCount + 1
	end)
end

watchForPlayerTurning()
```

![Roblox turn example using dot product](/_media/roblox-turn-detector.gif)

In this example, I print the angle, but you substitute that line for a function that changes the character's animation or plays a sound, and so on.