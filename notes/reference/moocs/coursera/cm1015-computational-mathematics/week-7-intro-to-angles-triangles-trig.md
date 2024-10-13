---
title: Week 7 - Intro To Angles, Triangles and Trigomonetry
date: 2023-05-29 00:00
modified: 2023-05-29 00:00
status: draft
---

* [[Angle]]
    * Measure of separation of 2 [Ray](../../../../permanent/ray.md) emanating from a vertex v.
    * Measured in degrees or [Radians](../../../../permanent/radians.md).
    * Types of angle
    * Degrees
        * 1 degree is $\frac{1}{180}$ of a flat angle.
        * Degree is a sexagesimal [Number Base](../../../../permanent/number-bases.md).
            * 1 minute is $\frac{1}{60}$ of a degree, 1 second of $\frac{1}{60}$ of a min
    * [Radians](../../../../permanent/radians.md)
        ![](../../../../journal/_media/week-7-radians.png)
    * $r \rightarrow 1 \text{ radian} \Rightarrow \text{ (circ) } 2\pi r \rightarrow 2 \pi \text{ radians }$
    * $360\deg = 2\pi \text{ radians }$
    * $\text{ radians } = \text{ degrees } \times \pi / 180$
* [[Triangle]]
    * A polygon with 3 sides.
    * ![](../../../../journal/_media/week-7-triangles.png)
        * Property #1: sum of angles is equal to 180*
            * $\alpha + \beta + \gamma = 180°$
        * Property #2: surface area = (side c * height) / 2
            * $S = c \times h / 2$
        * Property 3: perimeter is all side added together.
            * $P = a + b + c$
    * Types of triangle
        * Triangle: right triangle.
            * ![](../../../../journal/_media/week-7-right-triangle.png)
        * Isosceles.
            * 2 sides of the same length, which implies to adjacent angles are equal.
              ![](../../../../journal/_media/week-7-intro-to-angles-triangles-trig-isosceles.png)
        * Equilateral
            * All sides are equal to 60 degrees
                ![](../../../../journal/_media/week-7-intro-to-angles-triangles-trig-equilateral.png)
        * [[../../../../permanent/similar-triangles]]
            * Similar triangles rescale one (zoom in or out) and will coincide with the other.
            * Same angle, proportinal sides: AB/EG = AC/EF = BC/GF
                 ![](../../../../journal/_media/week-7-intro-to-angles-triangles-trig-similar.png)
 * [[Right Triangle]]
     * ![](../../../../journal/_media/week-7-intro-to-angles-triangles-trig-right-triangle.png)
     * SOH CAH TOA
     * Opposite / Hypotenuse = $\cos(90 - \theta) = Sin(\theta)$
         * SOH
     * Adjacent / Hypotenuse = $\sin(90 - \theta) = \cos(\theta)$h
         * CAH
     * Opposite / Adjacent = $\sin(\theta) \ cos(\theta) = \tan(\theta)$
         * TOA
     * [Pythagoras Theorem](../../../../permanent/pythagoras-theorem.md)
         * $a^2 + b^2 = h^2$
     * From 1) and 2) it follows: $h^2 \sin^2(\theta) + h^2 \cos^2(\theta) = h^2$
         * We can rewrite as: $\sin^2(\theta) + \cos^2(\theta) = 1$
     * $a = h \sin(\theta)$
     * $b = h \cos(\theta) = h \sin(90 - \theta)$
     * What if $\theta \rightarrow 0$?
         * Hypotenuse will slowly collapse on adjacent side. $h$ will coincide with $b$ and $a$ will go to 0.
         * This means: $\cos(\theta) = \sin(90) = 1$, $\sin(0) = \cos(90) = 0$
         * [[Sine rule]]
             * $a / \sin(\theta) = b / \sin(90 - \theta) = h = h / \sin(90)$
             * Applies to a general triangle:
                 ![](../../../../journal/_media/week-7-intro-to-angles-triangles-trig-sine-rule.png)
            * Generalised Pythagoras Theorem: $a^2 = b^2 + c^2 - 2bc \cos(\alpha)$ also known as [Law Of Cosines](../../../../permanent/law-of-cosines.md).

## Reading

Croft, A. and R. Davison Foundation maths. (Harlow: Pearson, 2016) 6th edition. Chapter 22 Introduction to trigonometry.
