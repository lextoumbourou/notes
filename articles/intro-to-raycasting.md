Link: https://developer.roblox.com/en-us/articles/Raycasting
Title: [[Intro to Raycasting]]
Author: [[Roblox]]
Type: #article
Topics: [[roblox]] [[raycasting in computer graphics]]

# Intro to Raycasting
* A tool used to send out invisible ray from a `Vector3` origin point to a specified direction, also represented as a `Vector3`
* Example local script to send a ray from the humanoid root part 100 studs down on the y-axis:

	```
    local humanoidRootPart = game.Players.LocalPlayer.Character:WaitForChild('HumanoidRootPart')
    local rayOrigin = humanoidRootPart.CFrame.Position
    local rayPosition = Vector3.new(0, -100, 0)
    local raycastResult = workspace:Raycast(rayOrigin, rayPosition)
    ```
    
* The distance between the original and direction is the functional length (magnitude) of the vector - nothing past it will be detected.
* If you have an unknown direction vector, but a known origin and destination, the destination can be calculated with simple algebra:
    * An unknown direction vector can be calculated using a known origin and destination, like this:
        * The origin + direction vector indicates the ray's destination.
            * ``rayOrigin + rayDirection = rayDestination``
        * The origin is subtracted from both sides of the equation:
            * ``rayOrigin + rayDirection - rayOrigin = rayDestination - rayOrigin``
        * The ray’s direction is the destination minus the origin:
            * ``rayDirection = rayDestination - rayOrigin``
* A `RaycastParams`  object can be passed that allows for part whitelisting or blacklisting, amongst other features.
*  An object with the following properties is returned on a successful hit (or `nil`):
    * `RaycastResult.Instance` - the `BasePart` or `Terrain` cell the ray intersects.
    * `RaycastResult.Position` - the world space point at which the intersection occurred which will be a point on the instance.
    * `RaycastResult.Material` - the Material at the intersection point.
    * `RaycastResult.Normal` - the normal vector of the intersected face.