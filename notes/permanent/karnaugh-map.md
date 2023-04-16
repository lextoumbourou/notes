---
title: Karnaugh Map
aliases: K-Map
date: 2022-12-24 00:00
status: draft
---

A Karnaugh Map or K-Map is a way to graphically represent [Boolean Function](Boolean%20Function)s.

At the time of writing, I can't describe them in words, only example.

If you have a 2 variable truth table as shown here:

| x   | y   | f   |
| --- | --- | --- |
| 0   | 0   | 0   |
| 0   | 1   | 0   |
| 1   | 0   | 1   |
| 0   | 1   | 1    |

You would represent it in a K-Map as follows:

|     |              | y'           | y            |
| --- | ------------ | ------------ | ------------ |
|     |              | $\mathbf{0}$ | $\mathbf{1}$ |
| x'  | $\mathbf{0}$ | 0            | 0            |
| x   | $\mathbf{1}$ | 1            | 1             |

It can be used for expressions with 2, 3, 4 or 5 variables.

If we have a 3 variable truth table, in this example the function is $f(x, y, z) = (x + y) . z$

| x   | y   | z   | f   |
| --- | --- | --- | --- |
| 0   | 0   | 0   | 0   |
| 0   | 0   | 1   | 0   |
| 0   | 1   | 0   | 0   |
| 0   | 1   | 1   | 1   |
| 1   | 0   | 0   | 0   |
| 1   | 0   | 1   | 1   |
| 1   | 1   | 0   | 0   |
| 1   | 1   | 1   | 1   |

The K-Map would looks like this:

|     | y'z' | y' z | y z' | y z |
| --- | ---- | ---- | ---- | --- |
| x'  | 0    | 0    | 0    | 1   |
| x   | 0    | 1    | 0    | 1    |

K-Maps can be used to simplify Boolean expressions.

What we do is find the maximum number of adjacent 1s to form a rectangle. The length of the rectangle must be a power of 2: 1, 2, 4, 8 etc.

In my 2nd example, the y z column is the longest rectangle of 1s. So I write: $yz$, then the next set of 1s is in the x row. Somehow, I end up with the sum of products form:

$yz + xz$
