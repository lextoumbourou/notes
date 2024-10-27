---
title: "Week 15: Limits and differentiation"
date: 2023-09-03 00:00
modified: 2023-09-03 00:00
status: draft
---

## Lesson 8.1. Definition of limit for a sequence

Example:

$a_n = n / (n+1)$

$0, \frac{1}{2}, \frac{2}{3}, \frac{3}{4}, ... \frac{100}{101}$

We can see from this that the sequence is approach 1. This is an example of a convergent sequence.

In general: $\lim_{n \rightarrow \infty} a_n = L \text{ if } \forall \epsilon > 0 \ \exists \ N: \text{ for } n > N |a_n -L| < \epsilon$

In other words, as you go further along the sequence, the terms get closer to L. You can pick any arbitrarily small value, and find somewhere on the sequence where the different between the value at the sequence, and the limit L is smaller than the epsilon value.

If limit exists and is finite, the sequence is convergent $a_n = \frac{n}{n+1}$ converges to 1.

If limit doesn't exist, the sequence is said to be divergent.

$a_n = 1 \times 3^n$ - $1, 3, 9, 27 ... 243 ...$ diverges to $\infty$

$a_n = \sin(\pi \frac{n}{2}) = 0. 1, 0, -1, 0, 1, 0, -1$ ... also does not converge as it oscillates between multiple values.

Some examples:

* $\lim_{n \rightarrow \infty} dn = \frac{1}{(n+1)^3}$
    * $1, \frac{1}{8}, \frac{1}{27}, \frac{1}{64}, ...$
    * We can see that the limit converges to 0.
* $\lim_{n \rightarrow \infty} dn = -2 + (-1)^m$
    * $-1, -3, -1, -3, -1, -3$
    * The limit is not convergent, as it oscillates.
* $\lim_{n\rightarrow \infty} \frac{n^2 + 2n + 5}{3n^2 + 2}$
    * One approach, find highest power and rewrite to line up:
        * $\frac{n^2(1 + \frac{2}{n} + \frac{5}{n^2})}{n^2(3 + \frac{2}{n})}$
        * $\frac{1 + \frac{2}{n} + \frac{5}{n^2}}{(3 + \frac{2}{n})}$
        * $\frac{1}{3}$
    * Limit converges to $\frac{1}{3}$

## Lesson 8.2. Limit and continuity of a function

$\lim_{x \rightarrow x0} f(x) = L$

$\text{if } \forall \ \epsilon > 0 \ \exists \ \epsilon > 0: \text{ for } |x -x_0| < epsilon \rightarrow |f(x) - L| < \epsilon$

Formalises the idea that if $f(x)$ gets arbitrarily close to value $L$, the closer you get to x0 then the function is L.

If limit is finite and coincides with the value of the function in x_0, i.e. if $f(x_0) = lim_{x \rightarrow x0} \ f(x) = L$

the function is said to be a [Continuous Function](Continuous%20Function) in $x_0$, if the value exists and is its limit, and the limit is finite.

A function is continuous, if you can draw it with a pencil without living the pencil.

Example:

$f(x) = 4-3x^2$ calculate $\lim_{x \rightarrow 1} f(x)$

Shown as a table:

| x=0.5       | 0.9  | 0.99 | 0.999 | 1   | 1.001 | 1.01 | 1.1 |
| ----------- | ---- | ---- | ----- | --- | ----- | ---- | --- |
| f(x) = 3.25 | 1.57 | 1.06 | 1.006 | ?   | 0.993 | 0.94 | 0.37    |

Importantly the limit must be the same from both sides.

Limit exists if and only if $\lim_{x \rightarrow x_0^{-}} f(x) = \lim_{x \rightarrow x_0^{+}} f(x) = L$

Here, we can see that the limit is 1 from either side.

[Discontinuous Function](Discontinuous%20Function)

$$
y = f(x) = \begin{equation}
\left\{
    \begin{array}{lr}
        x & x \geq 0\\
        1 - x & x < 0\\
    \end{array}
\right\}
\end{equation}
$$

$\lim_{x \rightarrow 0} f(x)$?

Shown as a table:

| x=-1       | -0.5  | -0.1 | -0.01 | 0   | 0.01 | 0.1 | 0.5 |
| ----------- | ---- | ---- | ----- | --- | ----- | ---- | --- |
| f(x)=2 | 1.5 | 1.1 | 1.01 | ?   | 0.01 | 0.1 | 0.5    |

Can see from the left, we're approaching 1, and from the right, approach 0. So the limit does not exist. Therefore, we say f not continuous in x=0.

Another interesting case of discontinuous function. You can have a limit that exists, but is different from the value at the point. It means that f is not continuous in x_0 = 0, even though the limit exist and is 1.

![](/_media/week-15-limits-and-differentiation-limit-not-continuuos.png)

$\lim f(x) = 0 = \lim f(x) \neq f(0) = 1$

## Lesson 8.3 Derivative of a function

[Derivative](../../../../../../permanent/derivative.md)

Directly connected to the concept of slope or gradient of a function

Consider the slope of a [Linear Function](../../../../permanent/linear-function.md).

Straight line $y = f(x) = mx + k$, where $m = \tan \alpha$

$\tan \alpha = \triangle y / \triangle x$

Consider a more generic function f(x):

![](/_media/week-15-limits-and-differentiation-generic-func.png)

The slope at any point x, the line tangent to the curve is the slope.

![](/_media/week-15-limits-and-differentiation-tangent.png)
The derivative is the slope of the line tangent to the curve at a point P.

Derivative;

$f'(x) = \frac{df}{dx} = \lim_{\triangle x \rightarrow 0} \frac{f(x + \triangle{x}) - f(x)}{\triangle x}$

[Derivative](../../../../../../permanent/derivative.md) from first principles.

* $f(x) = x$
    * $f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h} = \frac{x + h - x}{h} = \frac{h}{h} = 1$
    * $f'(x) = \frac{d(x)}{dx} = 1$
* f(x) = 1/x
    * $f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}$
    * $f(x+h) = \frac{1}{x+h}$
    * $\frac{f(x+h) - f(x)}{h} = \frac{\frac{1}{x+h} - \frac{1}{x}}{h} = \frac{\frac{x - (x + h)}{(x + h) x}}{h} = \frac{-1}{x(x+h)}$
    * $\lim_{h \rightarrow 0} = - \frac{1}{x(x+h)} = -\frac{1}{x^2} \Rightarrow f'(x) = -\frac{1}{x^2}$

## Essential Reading

Croft, A. and R. Davison, Foundation maths. (Harlow: Pearson, 2016) 6th edition. Chapter 34 and 35.

[Gradient Function](Gradient%20Function)

If we have a function $y = f(x)$ and want its slope, or gradient at sevenl points.

See the function $y=2x^2 + 3x$. At differnet points the slope of the graph is diff
![](../../../../journal/_media/week-15-limits-and-differentiation-slope-points.png)
To find the exact gradient at a curve, need gradient function.

Writteen as $\frac{dy}{dx}$ read as dy by dee x. Simplified to y' (y prime/dash).

For a function of form $y = x^{n}$ gradient is found from formula: $y' = nx^{n-1}$

Examples:
find the gradient of
(a) $y = x^3$
$y' = 3x^2$

(b) $y = x^4$
$y' = 4x^3$

(c) $y = x^2$
$y' = 2x$

(d) y = x
$y' = x^{0} =1$

Now you can find the gradient of the graph. The graidnet atany value of x is found by subing the value into gradient function.

Examples:
Find the gradient of y = x^2 at point:

a) x = -1
y' = 2x = -2 (falling)

b) x = 0
y' = 0 (flat)

c) x= 2
y' = 4 (rising)

d) x = 3
y' = 6 (rising)

If the gradient is negative, the curve is falling. If positive, it's rising. The size of the gradient is a measure of how rapidly the fall or rise is taking place.

Common gradient functions:

| y = f(x)              | y' = f'(x)   |
| --------------------- | ------------ |
| constant              | 0            |
| x                     | 1            |
| $x^2$                 | 2x           |
| $x^n$                 | $nx^{n-1}$   |
| $e^x$                 | $e^x$        |
| $e^{kx}$ (k=constant) | $ke^{kx}$    |
| $\sin x$              | $\cos x$     |
| $\cos x$              | $-\sin x$    |
| $\sin kx$             | $k \cos kx$  |
| $\cos kx$             | $-k \sin kx$ |
| ln kx                 | 1/x             |

The gradient function is also called the **first derivative** or just **derivative**.

Process of obtaining is called **differentiation**. Being asked to **differentiate** y = x^5, is being asked to find its gradient function (which is $5x^4$).

Since it measures how fast the graph is changing, it's also referred to as **rate of change** of y.

The area of study concerned with differentiation is known as differential calculus.

34.6 Find gradient function y' when y is:

a) $\sin x$

$y' = \cos x$

b) $\sin 2x$

$y' = 2 \cos 2x$

c) $\cos3x$

$y' = -3 \sin 3x$

d) $e^x$

y' = e^x

34.7 Find the gradient function of $y =e^{-x}$. Hence find the gradient of the graph of $y$ at the point where $x = 1$.

Can rewrite as $e^{-1x}$

$-1e^{-x}$

Where x = 1

$-e^{-1} = -0.368$

34.8 Find the gradient function of $y = \sin 4x$ where $x = 0.3$

$y' = 4 \cos 4x$
$y' = 4 \cos 4 (0.3)$
$y' = 1.4494$

Exercise 34.2

34.3 Some rules for finding gradient functions

Rule 1: If $y = f(x) + g(x)$ then $y' = f'(x)+ g'(x)$

Example: find the gradient function of $y = x^2 + x^4$

${x^2}^{\prime} = 2x$
${x^4}^{\prime} = 4x^3$
$y' = 2x  + 4x^3$

Rule 2: If $y = f(x) - g(x)$ then $y' = f'(x) - g'(x)$

Example: find the gradient function of $y = x^5 - x^7$

${x^5}^{\prime} = 5x^4$
${x^7}^{\prime} = 7x^6$
$y' = 5x^4 - 7x^6$

Rule 3: if $y =kf(x)$, where k is a number, then $y' = kf'(x)$

Find the gradient function of $y = 3x^2$

$y' = (3) 2x$
$y' = 6x$

Examples:

34.12 Find derivative of $y = 4x^2 + 3x^{-3}$

${4x^2}^{\prime} = (4)2x = 8x$
${3x^-3}^{\prime} = (3) -3x^{-4} = -9x^{-4}$
$y' = 8x - 9x^{-4}$

34.13 Find the derivative of:

a) $y = 4 \sin t - 3 \cos 2t$

$4 \cos t + 6 \sin 2t$

b) $y = \frac{e^{2t}}{3} + 6 + \frac{ln(2t)}{5}$

Can rewrite $y = \frac{1}{3} e^{2t} + 6 + \frac{1}{5} ln(2t)$

$y' = \frac{2}{3}e^{2t} + \frac{1}{5} \frac{1}{t} = \frac{2e^{2t}}{3} + \frac{1}{5t}$

Exercise 34.3 (to do)

### Lesson 34.3 Higher derivatives
