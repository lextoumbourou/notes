---
title: Week 3 - Sequences and Series
date: 2023-04-22 00:00
modified: 2023-04-22 00:00
status: draft
---

## Lesson 2.1 Introduction to sequences and series

* When folding paper, every fold the thickness of the paper increases:
    * $0.1$mm
    * $0.2$mm
    * $0.4$mm
    * ...
    * $0.1 \times 2^n$mm
    * If you fold 42 times:
        * $0.1 \times 2^{42}$mm = ~440000 km
        * Which is a bit over the distance to the moon: 384400 Km.
    * This is an example of a Geometric Sequence.
* Sequence
    * $a, a \times q, a \times q^2, ..., a \times q^n$
    * Formal definition:
        * Given a [Set](permanent/set.md) X, a sequence is a [[Function]] $a: N \rightarrow X$
            * i.e. a set a(0), a(1), ...., a(n) ... denoted with $\{a_n\}_{n \in N}$
    * Can be defined explicitly $a_n = f(n)$
        * Example: $a_n = 2n + 1 \rightarrow 1, 3, 5, 7, 9 ...$
    * Or by recursion i.e. $a_n = f(a_{n-1}, ..., a_{n-k}$
        * Example 1: Arithmetic sequence $a_n = q+ a_{n-1}$
            * $\rightarrow a_0, a_0 + q, a_0 + 2q, ... \rightarrow a_n = n \times q + a_0$
        * Example 2: Geometric sequence $a_n = q \times a_{n-1}$
            * $\rightarrow a_0, a_0q, a_0q^2, a_0q^3, ...., a_0q^{n},... \rightarrow a_n = a_{0} \times q^{n}$
    * A sequence is said to be **convergent**, when upon increasing $n$ it approaching a finite constant value
        * $\lim_{n\rightarrow\infty} a_n = L < \infty$
        * Example:
            * Geometric sequence when q < 1
                * q = 1/2, 1/4, 1/4, 1/8, ...., 1/64, .... converges to 0.
    * On the flip side, a sequence is said to be divergent when increasing $n$ it never reaches a constant finite value (either goes to $\infty$ or oscillates)
* Fibonacci Sequence:
    * Most common definition by recursion: $a_0 = 0, a_1 = 1, a_n = a_{n-1} + a_{n-2}$
        * 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144
    * One of the reasons for this sequence importance, is that the ratio between a sequence value and its previous value is known as the [[Golden Ratio]].
        * $\frac{a_n}{a_{n-1}} \rightarrow \phi = \frac{1 + \sqrt{5}}{2} = 1.618$
* Fibonacci Spiral
    * If you take the sequence of squares whose sides are given by the number of an element in the sequence.
    * If you join 2 opposite sides of the square with an arccos circumference, you obtain the Fibonacci spiral, which is observed throughout nature.
        ![Fibonacci Spiral](../../../../journal/_media/week-3-sequences-and-series-fib-spiral.png)

## Essential Reading / Topic 2       

Croft, A. and R. Davison Foundation maths. (Harlow: Pearson, 2016) 6th edition. Chapter 12 Sequences and series. 

* Sequence
    * A sequence is a set of numbers written down in a specific order:
        * 1, 3, 5, 7, 9
        * -1, -2, -3, -4
    * There doesn't need to be a rule that relates numbers in the sequence.
    * Every number in the sequence is called a **term** of the sequence.
        * The "number of terms" in the first sequence is five, and the number of terms in the 2nd is four.
    * We can use `...` to indicate the sequence continues.
        * Sequences that have a finite number of terms are **finite sequences**
        * Sequences that have an infinite number are **infinite sequences**
* Notation
    * Subscript notation is used for different terms in a sequence.
        * For the sequence 1, 3, 5, 7, it would be:
            * $x_{1} = 1, x_{2} = 3, x_{3} = 5, x_{4} = 7$
* Arithmetic progressions
    * When you calculate a sequence by adding a fixed amount to the previous term, it's called an **arithmetic progression** or **arithmetic sequence**.
        * The fixed amount added each time is **common difference**.