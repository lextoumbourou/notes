Link: https://lodev.org/cgtutor/
Title: [[Lode's Computer Graphics Tutorial]]
Author: [[Lode Vandevenne]]
Type: #article
Topics: [[graphics]] [[old school]]

# Lode's Computer Graphics Tutorial

## Raycasting

* Raycasting is a technique for rendering basic 3d spaces from a 2d made famous from the Wolfenstein 3d rendering engine.
* Define map as a series of 2d squares, giving squares either 0 (nothing in the square) or a positive number to represent a texture in each square.
    ![map as 2d grid](raycasting-2d-grid.png)
* Break the players screen up into vertical slices.
* For each slices, cast a [[Ray]] from the player's location in directions depending on the player's looking direction and their field of view (FOV).
    ![map as 2d grid with player](raycasting-2d-grid-with-player.png)
* At each point on the [[Ray]] determine if the ray hits a wall. If it does, use that to determine how far away the wall is and thus how high on the screen it should be rendered.
* Since early computers couldn't calculate the hits for an infinite number of points, a grid with fixed squares sizes was used and an algorithm called Digital Differential Analysis (DDA) to minimise the number of hit checks required.