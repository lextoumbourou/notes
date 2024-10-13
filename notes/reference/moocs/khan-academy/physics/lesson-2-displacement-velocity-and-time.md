---
title: "Lesson 2: Displacement Velocity And Time"
date: 2023-08-13 00:00
modified: 2023-08-13 00:00
status: draft
---

Notes from [Lesson 2: Displacement, velocity, and time](https://www.khanacademy.org/science/physics/one-dimensional-motion/displacement-velocity-time) on Khan Academy.

## Intro to vectors and scalars

* [Vector](../../../../permanent/vector.md) has a magnitude/size AND distance. A [Scalar](Scalar) only has a distance.
* [Displacement](../../../../journal/permanent/displacement.md)
    * In physics, we call distance "displacement".
    * Displacement is a vector quantity.
* Speed
    * If we travelled 5 meters over 2 seconds (change in time is 2 seconds), we can express it as:
        * $\frac{\text{5 meters}}{\text{2 seconds}}$ = 2.5 metres / second
            * This is called "speed".
            * This is a scalar value.
* Velocity
    * $\frac{\text{5 meters to the right}}{\text{2 seconds}}$ = 2.5 metres / second to the right
    * When you include the direction and the displacement, you are talking about velocity.

## Introduction to reference frames

* [Frame of Reference](../../../../journal/permanent/frame-of-reference.md)
    * Point of view for which you're measuring from.

    ![lesson-2-displacement-velocity-and-time-frame-of-ref](../../../../journal/_media/lesson-2-displacement-velocity-and-time-frame-of-ref.png)

## What is displacement

* To describe an object's motion, we first need to describe its position.
    * Specifically: need to specify its position relative to a convenient [Frame of Reference](../../../../journal/permanent/frame-of-reference.md).
        * Earth is a common reference frame.
        * If a person is in an airplane, we might use that as the frame of reference.
* Variable $x$ is commonly used to represent the horizontal position.
* Variable $y$ is used to represent the vertical position.
* Change in position is known as [Displacement](../../../../journal/permanent/displacement.md).
* Mathematically defined as follows:
    * Displacement = $\triangle x = x_f - x_0$
        * x_f refers to the value of the final position.
        * x_0 refers to the value of the initial position.
        * $\triangle x$ is the symbol used to represent displacement.
* Displacement is a [Vector](../../../../permanent/vector.md), which means it has a direction and magnitude and is represented visually as an arrow that points from initial position to final position.
* Distance vs Distance travelled
    * Distance is defined to be the magnitude or size of displacement between 2 positions.
    * Distance travelled is the total length of the path travelled between 2 positions.
        * Not a vector.
        * No direction.
    * Distance travelled is not necessarily the magnitude of displacement (i.e. distance between two points).
        * For example, if an object changes direction in its journey, the total distance travelled will be larger than the magnitude of the displacement between 2 points.
* In kinematics, we are mostly interested in displacement and magnitude of displacement, rarely distance travelled.
* It's common to forget to include a negative sign, when needed, with displacement.
    * Be sure to subtract the init position from the final position.

## Calculating average velocity or speed

* If Shantanu was able to travel 5 km north in 1 hour in his car, what was his average velocity?
    * Velocity is displacement over time.
    * $\vec{s}$ commonly used for displacement in physics.
    * $\vec{v} = \frac{\vec{s}}{t}$
    * $\vec{v} = \frac{\text{5km north}}{\text{change in time}}$
    * $\vec{v} = \frac{\text{5km north}}{\text{1 hour}}$
    * $\vec{v} = 5 \frac{\text{km}}{\text{hour}} \text{north}$

## Solving for time

* Ben is running at a constant velocity of 3 m/s to the east. How long will it take him to travel 720 meters?

3 m/s = 720m / t
3 m/s * t = 720 m
t = 720 / 3
t = 240 s

* If Maria travels for 1 minute at 5m/s to the south, how much will she be display?

v = s /t
5 m/s south = s / 1min
5 m/s south * 1m = s
s = 5m/s south

## Instantaneous speed and velocity

* Instantaneous Speed
    * Speed of an object at a particular moment in time.
    * If you include the direction, you get instantaneous velocity.
        * Note that it's different from average velocity.
* Issac Newton figured out a method of calculating instantaneous velocity, which went on to be called calculus.
* Methods of calculating instantaneous velocity:
    * First method: if your velocity never changes, then average velocity gives you instaneous:
        * $V_{\text{average}} = \triangle x / \triangle y = v_{\text{instantaneous}} = 7m/s$
    * Slope at any point gives you instanneous.
    * If acceleration is constant, we can use the kinematic formulas:
        * v = v; + at
        * v^2 = v_^2 + 2a\triangle x$
