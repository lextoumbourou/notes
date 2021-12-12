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
task.wait(1)
local TweenService = game:GetService('TweenService')

while true do
	local goal = {}
	goal.Position = Vector3.new(
		math.random(1, 5),
		math.random(1, 5),
		math.random(1, 5)
	)
	goal.Orientation = Vector3.new(
		math.random(1, 360),
		math.random(1, 360),
		math.random(1, 360)
	)
	local tweenInfo = TweenInfo.new(5, Enum.EasingStyle.Linear)
	local tween = TweenService:Create(part, tweenInfo, goal)
	tween:Play()
	task.wait(5)
end
