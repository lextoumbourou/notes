---
title: Inverse Trigonomentric Functions
date: 2023-08-11 00:00
modified: 2023-08-11 00:00
status: draft
---

For each of the 3 trigonometric functions, there is a function that can invert the results.

For example, the [[../journal/permanent/sine]], can be expressed as follows

$\sin(\theta) = x$

Is there a function that can take $x$ and return $\theta$? It's called $\arcsin$

$\arcsin(x) = \theta$

We have the following functions:

* $\arcsin(x)$ or $\sin^{-1}(x)$ is the inverse of $\sin(x)$
* $\arccos(x)$ or $\cos^{-1}(x)$ is the inverse of $\cos(x)$
* $\arctan(x)$ or $\tan^{-1}(x)$ is the inverse of $\tan(x)$

Trig functions aren't really invertible (see [Inverse Function](inverse-function.md)) as they have multiple inputs with the same output (see [Injective Function](injective-function.md)), like $\sin(0) = \sin(\pi) = 0$.

So how can we define $\sin^{-1}(0)$?

To do that, we have to restrict the domain of the original functions to an interval where they are invertible. These domains determine the range of inverse functions.

The value from the appropriate functions is called the **Principal value** of the function:

| Function | Radians                                                | Degrees |
| -------- | ------------------------------------------------------ | ------- |
| $\arcsin$ | $-\frac{\pi}{2} \le \arcsin(\theta) \le \frac{\pi}{2}$ |    $90° \le \arcsin(\theta) \le 90°$     |
| $\arccos$ | $0 \le \arccos(\theta) \le \pi$ |    $0° \le \arccos(\theta) \le 180°$     |
| $\arctan$ | $-\frac{\pi}{2} \lt \arctan(\theta) \lt \frac{pi}{2}$ |    $-90° \lt \arctan(\theta) \lt 90°$     |

If the theta is outside of
