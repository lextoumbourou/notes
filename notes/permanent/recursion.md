---
title: Recursion
date: 2023-04-09 00:00
cover: /_media/recursion-tree.png
hide_cover_in_article: true
summary: a method of solving a problem where a function calls itself
---

**Recursion** is a method of solving problems where a function calls itself to solve a smaller instance of the problem. A recursive solution requires a [Base Case](../../../permanent/base-case.md) to avoid infinite recursion, ensuring that the calls eventually terminate.

All recursive problems can be solved using iteration; however, some algorithms, particularly a [Divide-and-Conquer Algorithm](divide-and-conquer-algorithm.md), can be solved much more elegantly with recursion.

## Fibonacci numbers example

The Fibonacci number function is a recursive function that calculates the sum of the previous two numbers: $F_n = F_{n - 1} + F_{n - 2}$

The base case is $\text{Fibonacci}(1) = 1$, $\text{Fibonacci}(0) = 0$

In pseudocode:

```javascript
function Fibonacci(n)
     assert n >= 0

    if n < 2 then
        return n
    end if

    return Fibonacci(n-1) + Fibonacci(n-2)

end function
```

We visualise a call to $\text{Fibonacci}(5)$ by representing the call stack as a tree, like this:

![Recursion Tree](../_media/recursion-tree.png)
*[Source dhovemey at ycp](http://faculty.ycp.edu/~dhovemey/fall2005/cs102/lecture/fib5.png) (page now unavailable)*

Each branch opens two new branches until finally we reach the bottom of the tree and can propagate the answers back to the top.

## Factorial example

The Factorial algorithm is:

$n! = n \times (n - 1) \times (n - 2) \times ... \times 2 \times 1$

Which can be expressed recursively simply as:

$n! = n \times (n - 1)!$

The base case for Factorial comes at $0! = 1$

We can rewrite the function in pseudocode as follows:

```javascript
function Factorial(n)
    if n = 0 then
        return 1
    end if

    return n x Factorial(n - 1)

end function
```
