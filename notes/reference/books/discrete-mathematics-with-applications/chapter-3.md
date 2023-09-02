---
title: "Discrete Maths with Applications: Chapter 3"
date: 2023-04-15 00:00
modified: 2023-04-15 00:00
status: draft
---

## Chapter 3. Functions and Matrices

### 3.1 Concept of a function

[Function](../../../permanent/function.md) is fundamental and plays a role of unifying thread in every branch of maths. (pg 117.)

---

Let $X$ a $Y$ by any 2 nonempty sets.

A function $X$ to $Y$ is a rule that assigns each element $x \in X$ a unique element $y \in Y$.

![Function mapping example](../../../journal/_media/chapter-3-function-mapping-example.png)

Usually denoted by letters $f$, $g$, $h$, $i$, if $f$ is a function $X$ to $Y$ we write $f : X \rightarrow Y$.

The set $X$ is the **domain** of the function $f$ and $Y$ is the codomain of $f$, denoted by dom(f) and codom(f).

If $X = Y$ then f is a function on $X$.

One requirement of a function, is that every element in the domain is paired with a value in the output domain. $y$ is the **value** or **image** of the function. x is the **pre-image** of $y$ under $f$. $y$ is also known as **output** to the **input** (or **argument**) x.

pg 118, 119

---

In notation, $f$ is considered the function and $f(x)$ a value.

pg 120

---

A [Piecewise Function](../../../permanent/piecewise-function.md) is a function defined in terms of multiple formulas, think if-else statements in programming.

Example, let A = {0, 1..., 127}, a set of ordinal numbers in ASCII. Let f : A -> ASCII be defined b:

$$
\begin{equation}
f(n) = 
\left\{
    \begin{array}{lr}
        \text{nonprintable control character} & \text{if } 0 \le n \le 31 \text{ or } n = 127\\
        \text{uppercase letter} & \text{if } 65 \le n \le 90\\
        \text{lowercase letter} & \text{if } 97 \le n \le 122\\
        \text{other printable character} & \text{otherwise }\\
    \end{array}
\right\}
\end{equation}
$$

pg 121, 122

---

The geometric representation of a function, called a **graph**, is used to study functions.

pg. 123

---
