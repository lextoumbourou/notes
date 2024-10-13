---
title: Essense of linear algebra
date: 2021-04-06 00:00
category: reference/videos
cover: /_media/vectors-3blue1brown-cover.png
status: draft
parent: essence-of-linear-algebra
---

Notes from 3Blue1Brown video [Vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs) from the [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) series.

Vectors are the building block of linear algebra.

3 perspectives of vectors:

* Physics student
    * Vectors are arrows pointing in space defined by length & direction.
    * You can move a vector around and it remains the same vector.
    * If the space is a flat plane, it's a 2d vector.
* Computer scientist (or data scientist)
    * An orderer list of numbers.
    * Example, a house can be model as the vector: ```[square footage, sale price]```
  * Mathematician
    * A vector can be anything that has a notion of adding two together and multiplying a vector by a number.
* In Linear Algebra, the vector is nearly always rooted at the Origin, which is the place where x and y intersect on the coordinate system.
    * Note how it differs from physics student's perspective that can consider a vector at any point in space.
* The "back and forth" between the physics and computer science student understanding of vectors is that we represent coordinates for a line as lists of numbers.
* 2d coordinate system refresher
    * Has a horizontal line called the x-axis
    * Have a vertical line called the y-axis
    * Place where they intersect is called the Origin

         ![origin in coordinate system](/_media/origin-in-coordinate-system.png)

        * Can be thought of as the center of space and root of all vectors.
    * An arbitrary length is chosen to represent one, then tick marks are created on each axis spaced at this distance

        ![Tick marks in the coordinate system](/_media/tick-marks-in-coordinate-system.png)

* Coordinates of vector give instructions to get from the Origin to its tip

    ![Coords of vector](/_media/coords-of-vector.png)

    1. The first number describes how far to walk along the x-axis.
    2. The second number describes how far along the y-axis.

* We distinguish vectors from points by writing them vertically using square bracket notation.
* In 3d, an additional axis is drawn that's perpendicular to the other axes, adding a 3rd coordinate

    ![Vector in 3d space](/_media/vector-in-3d-space.png)

* Vector addition from the physics perspective:

    1. First draw both vectors:

        ![Vector addition step 1](/_media/vector-addition-step-1.png)

    2. Move the 2nd vector up to the tip of the tail of the first vector:

        ![Vector addition step 2](/_media/vector-addition-step-2.png)

    3. Draw a new vector that starts from the tail of the first to the tip of the 2nd. The new vector represents the sum of 2 vectors.

        ![Vector addition step 3](/_media/vector-addition-step-3.png)

* The above definition of vector addition is an extension to how we teach kids to add numbers using the number line: we start at a number, then step in the direction of the number we're adding.

    ![Vector addition analogy to number line addition](/_media/addition-along-number-line.png)

* The same operation written from the computer scientist perspective:

    $$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 + 3 \\ 2 + -1 \end{bmatrix} = \begin{bmatrix} 4 \\  1\end{bmatrix}$$

* Vector multiplication is when you multiply a vector by a number ([Vector Scaling](../../../permanent/vector-scaling.md)) either stretching or squishing the vector by that amount.

    ![Vector scaling](/_media/vector-scaling.png)

    * Referred to as "scaling the vector," hence the number in this context is called a "scalar."
    * Computer science perspective:

        $$2 \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 2 * 1 \\ 2 * 2 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$$

#Maths/LinearAlgebra/Vectors
