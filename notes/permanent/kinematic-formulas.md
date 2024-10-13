---
title: Kinematic Formulas
date: 2023-08-14 00:00
modified: 2023-08-14 00:00
status: draft
---

The [Kinematics](Kinematics.md) formulas are a set of equations that describe key properties of movement, under constant acceleration.

They are concerned with 5 key variables:

1. $s$ - describes the [Displacement](displacement.md) of an object: a vector quantity that describes how much an object moved. Or the delta between starting position and final position. Sometimes expressed as $\triangle x$
2. $t$ - Time elapsed.
3. $u$ - Initial velocity.
4. $v$ - Final velocity.
5. $a$ - [Acceleration](acceleration.md) (a constant in these formulas)

Also called [SUVAT Equations](../journal/permanent/suvat-equations.md).

If we know three of the five kinematic variable, we can use one of the kinematic formulas to solve for an unknown variable.

There are four key equations:

1. $s = \frac{u+v}{2} t$

Displacement is equation to the average velocity $\times$ time elapsed. The $\frac{u + v}{2}$ part is the average velocity under constant acceleration.

Note that $a$ is not included in this equation.

2. $v = u + at$ or $v - u = at$

Final velocity is the sum of the initial velocity plus the product of constant acceleration ($a$) over time $(t)$.

Note that $s$ is not included in this equation.

3. $s = ut + \frac{1}{2} at^{2}$

Displacement of an object is the amount the object would have moved at constant velocity $(u)$ for time $t$ plus, the additional displacement due to acceleration $\frac{1}{2} at^2$.

Note that $v$ is not included in the equation.

4. $v^2 = u^2 + 2as$ or $v^2 - u^2 = 2as$

Expresses the relationship between final velocity, initial velocity, constant acceleration and displacement of an object.

Note that $t$ is not included in this equation. So useful when time is not known.

---

These formulas only work if the acceleration is constant during the time interval. However, one of the most common forms of motion, free fall, happens under constant acceleration gravity.

The magnitde of acceleration due to gravitiy is $g = 9.81 \frac{m}{s^2}$

---

The important part of solving these kinematic formulas is to find which variable was not provided and which variable needs to be solved. Usually problems will include code words, that can be substituted for variables:

* starts from rest, dropped, means the initial velocity is 0: $u =0$
* comes to a stop, means the final velocity is 0: $v = 0$

Some problems will describe a free-falling projectile, which tells you that the magnitude of acceleration due to gravity is $g = 9.81 \frac{m}{s^2}$, so acceleration isn't provided in this case.
