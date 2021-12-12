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
local TweenService = game:GetService('TweenService')

local goal = {}
goal.Orientation = Vector3.new(
	part.Orientation.X,
	360,
	part.Orientation.Z
)
local tweenInfo = TweenInfo.new(5, Enum.EasingStyle.Linear, Enum.EasingDirection.Out, -1, false, 0)
local tween = TweenService:Create(part, tweenInfo, goal)
tween:Play()
tween.Completed:Wait()
