---
title: Piecewise Function
date: 2023-04-15 00:00
modified: 2023-04-16 00:00
status: draft
---

A function defined in terms of multiple formulas, think if-else statements in programming.

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
