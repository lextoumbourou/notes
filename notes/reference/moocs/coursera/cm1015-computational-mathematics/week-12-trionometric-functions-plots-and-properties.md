---
title: week-12-trionometric-functions-plots-and-properties
date: 2023-08-06 00:00
modified: 2023-08-06 00:00
status: draft
---

## Trigonometric functions, plots and properties

* Properties: $\sin(x)$
    * $\sin(180° - \alpha) = \sin(\alpha)$

    ```python
    assert math.sin(math.radians(180) - math.radians(30)) == math.sin(math.radians(30))
    ```

* $\cos(180° - \alpha) = -\cos(\alpha)$

    ```python
    >>> math.cos(math.radians(180) - math.radians(30))
    -0.8660254037844387
    >>> math.cos(math.radians(30))
    0.8660254037844387
    ```

    * $\sin(180° + \alpha) = -\sin(\alpha)$
    * $\cos(180° + \alpha) = -\cos(\alpha)$
    * $\sin(360° - \alpha) = -\sin(\alpha)$
    * $\cos(360° - \alpha) = \cos(\alpha)$
* Can use this relation to convert angle to the first quadrant before taking $\sin$.
* General rule:
    * $\cos(2n\pi + x) = \cos(x)$
    * $\sin(2n\pi + x) = \sin(x)$
* $\sin$ is a [Periodic Function](../../../../permanent/periodic-function.md) with a period that's $2\pi$.
     ![](../../../../journal/_media/week-12-trionometric-functions-plots-and-properties-2pi.png)

* $\cos$ has a similar curve, but shifted $\frac{\pi}{2}$
    ![](../../../../journal/_media/week-12-trionometric-functions-plots-and-properties-cos.png)

| Func     | period | frequency       | amplitude |
| -------- | ------ | --------------- | --------- |
| $\sin x$ | $2\pi$ | $\frac{1}{2\pi}$ | 1         |
| $\cos x$ | $2\pi$ | $\frac{1}{2\pi}$ | 1         |

* Properties: Tan(x)
    * Defined as ratio between sin and cos: $tan(x) = sin(x) / cos(x)$
    * In general: $\tan(2n\pi + x) = \tan(x)$

    ![](../../../../journal/_media/week-12-trionometric-functions-plots-and-properties-tan-1.png)
* Also: $\tan(x - \pi) = \tan(x)$
* $\tan(x + \pi) = tan(x)$
* Tangent has period $\pi$
    * You can add multiple instances of pi and leave unchanged: $\tan(n\pi + x) = \tan(x)$
* Amplitude of $\sin$
    * $f(x) = 2 \sin x$
        * Has the same period as $f(x) = \sin x$
            ![](../../../../journal/_media/week-12-trionometric-functions-plots-and-properties-sin-period.png)

| Func       | period | frequency        | amplitude |
| ---------- | ------ | ---------------- | --------- |
| $\sin x$   | $2\pi$ | $\frac{1}{2\pi}$ | 1         |
| $2 \sin x$ | $2\pi$ | $\frac{1}{2\pi}$ | 2         |

* $f(x) = \cos(2x)$

| Func       | period | frequency        | amplitude |
| ---------- | ------ | ---------------- | --------- |
| $\cos x$   | $2\pi$ | $\frac{1}{2\pi}$ | 1         |
| $2 \cos x$ | $\pi$ | $\frac{1}{\pi}$ | 1         |

![](../../../../journal/_media/week-12-trionometric-functions-plots-and-properties-amplitude-cos.png)
