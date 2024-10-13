---
title: Week 3 - Sequences and Series
date: 2023-04-22 00:00
modified: 2023-04-22 00:00
status: draft
---

#BscUoL/CM #Maths/Sequences

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
        * Given a [Set](permanent/set.md) X, a sequence is a [function](../../../../permanent/function.md) $a: N \rightarrow X$
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
    * One of the reasons for this sequence importance, is that the ratio between a sequence value and its previous value is known as the [Golden Ratio](../../../../../../permanent/Golden%20Ratio.md).
        * $\frac{a_n}{a_{n-1}} \rightarrow \phi = \frac{1 + \sqrt{5}}{2} = 1.618$
* Fibonacci Spiral
    * If you take the sequence of squares whose sides are given by the number of an element in the sequence.
    * If you join 2 opposite sides of the square with an arccos circumference, you obtain the Fibonacci spiral, which is observed throughout nature.
        ![Fibonacci Spiral](../../../../journal/_media/week-3-sequences-and-series-fib-spiral.png)

## Essential Reading / Topic 2

Croft, A. and R. Davison Foundation maths. (Harlow: Pearson, 2016) 6th edition. Chapter 12 Sequences and series.

* [Sequence](../../../../permanent/sequence.md)
    * A sequence is a set of numbers written down in a specific order:
        * 1, 3, 5, 7, 9
        * -1, -2, -3, -4
    * There doesn't need to be a rule that relates numbers in the sequence.
    * Every number in the sequence is called a [Term (Sequence)](Term%20(Sequence)) of the sequence.
        * The "number of terms" in the first sequence is five, and the number of terms in the 2nd is four.
    * We can use `...` to indicate the sequence continues.
        * [Finite Sequence](Finite%20Sequence)
            * Sequences that have a finite number of terms are **finite sequences**
        * [Infinite Sequence](Infinite%20Sequence)
            * Sequences that have an infinite number are **infinite sequences**
* Notation
    * Subscript notation is used for different terms in a sequence.
        * For the sequence 1, 3, 5, 7, it would be:
            * $x_{1} = 1, x_{2} = 3, x_{3} = 5, x_{4} = 7$
* Examples:
        * $x_{1} = 1, x_{2} = 3, x_{3} = 5, x_{4} = 7$
    * The terms of a sequence x are given by $x_k = 2k + 3$. Write down the terms $x_1$, $x_2$ and $x_3$:
        * $x_1 = 2 \times 1 + 3 = 5$
        * $x_2 = 2 \times 2 + 3 = 7$
        * etc
    * What is a finite sequence? A sequence that has a finite number of terms.
    * Write down the first five temrs of the sequencs given by:
* [Arithmetic Sequence](Arithmetic%20Sequence) (pg. 12.2)
    * When you calculate a sequence by adding a fixed amount to the previous term, it's called an **arithmetic progression** or **arithmetic sequence**.
        * [Common Difference](Common%20Difference)
            * The fixed amount added each time is **common difference**.
        * [First Term](First%20Term)
            * The starting point is called the **first term**.
        * For example, start at 1 and add 6 each time: 1, 7, 13, 19, ...
            * 1 is first term
            * 6 is common difference
    * Write down the four terms of arithmetic progression has first term 10 and common diff 3:
        * 10, 13, 16, 19, 22
    * Write down six terms of the arithmetic progression that has first 5 term 5 and comon diff -2
        * 5, 3, 1, -1 ...
    * Gneral notation for arithmetic progression:
        * first term $a$, common difference $d$. Second term is $a + d$. Third term is $a + d + d$, or $a + 2d$
    * Can write arithmetic progression as:
        * $a, a + d, a + 2d, a + 3d, ...$
        * a is first term, d is common difference.
    * Pattern:
        * first term is $a$
        * second term is $a + d$
        * third term is $a + 2d$
    * Therefore, formula: $a + (n- 1)d$
    * Example:
        * Find the 10th term of arithmetic progression with first term 3 and common diff 5.
            * $n = a + (n - 1)d = 3 + (9) \times 5 = 48$
    * Self assessment:
        * Arithmetic progress refers to a sequence that starts with a first term, and we progressively add a value called a **common difference**.
        * One sequence that is not an arithmetric progression: $4, 8, 9$
* [Geometric Progressions](../../../../permanent/geometric-progressions.md) (pg. 130)
    * When you multiply the previous term by a fixed amount.
    * The first value of the sequence: **first term**
    * The fixed amount to multiply previous term: **common ratio**
    * Examples:
        * Find first 6 terms of geometric progression with first term 3 and common ratio 2:
            * $3$, $2 \times 3 = 6$, $2 \times 3 \times 2 = 12, 24, 48, 96$
        * First 5 terms of geometric sequence with first term 5 and common ratio $\frac{2}{3}$
            * $5, 10/3, 20/9, 40/27, 80/81$
        * A geometric progression is given by 1, 1/2, 1/4, ... What is the common ratio?
            * First term is 1. What must you multiple to get 1/2? 1/2
            * Common ratio can always be found by dividing any term by the previous term.
        * First 6 terms of geometric sequence with first term 4 and common ratio -1.
            * 4, -4, 4, -4, 4, -4
            * Because the common ratio is negative, the terms of the sequence alternate in sign.
        * General notation for geometric sequences.
            * First term is $a$ and common ratio is $r$.
            * 2nd term is $a \times r$
            * Third term is $(ar) \times r = ar^2$
            * Fourth is $(ar) \times r \times r = ar^3$
        * A geometric progression can be written as: $a, ar, ar^2, ar^3, ...$
            * $a$ is the first term, $r$ is the common ratio.
        * Therefore, the $nth$ term is of a geometric progression is given by $ar^{n-1}$
        * Example: the 7th of the progression with first term 2 and common ratio 3:
            * $ar^{n-1} = 2 \times 3^{7-1} = 1458$
        * Explain what is mean by a geometric sequence:
            * A geometric sequence is a sequence that start with a first term, and the next term is creating by multiplying by some amount $r$.
            * $a$ is the first term
            * $r$ is the common ratio.
        * Give one example of a sequence that is a geometric sequence and one that is not.
            * $1, 2, \frac{1}{2}$ is not a geometric sequence.
            * $1, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}$
        * When can you say about the terms of a geometric progression when the common ratio is 1?
            * There will only be one unique term the first term.
        * What can you say about the terms of a geometric progression when the common ratio is -1?
            * The terms will oscillate between a negative and positive value.
        * 1. Write down the first 4 terms of a geometric sequence with first term 8 and common ratio 3.
            * $8$
            * $8 \times 3 = 24$
            * $8 \times 3^2 = 72$
            * $8 + 3^3 = 216$
        * 2. Write down the first four terms of the geometric progression with first term 1/4 and common ratio 4/5.
            * $\frac{1}{4}$
            * $\frac{1}{4} \times \frac{4}{5} = 0.2$
            * $\frac{1}{4} \times \frac{4}{5}^2 = 0.16$
            * $\frac{1}{4} \times \frac{4}{5}^3 = 0.128$
        * 3. A geometric sequence has first term $4$ and common ratio $2$
            * Find (a) on the 5th term and (b) on the 11th term.
                * $ar^{5-1} = (4)(2)^{5-1} = 64$
                * $ar^{11-1} = (4)(2)^{11-1} = 4096$
        * 4. A geometric progression is given by $2, -1, \frac{1}{2}, -\frac{1}{4}$
            * $-\frac{1}{2}$
* [Infinite Sequences](Infinite%20Sequences)
    * A sequence that continues indefinitely.
    * Use ... to indicate it: $1, 2, 3, 4, 5, ...$
    * You can also have a sequence where the terms get closer to a fixed value. Ie: $1, \frac{1}{2}, \frac{1}{3}, \frac{1}{4}, \frac{1}{5}, ...$
        * If you continue on forever, eventually the terms approach the value 0.
    * We can write the sequence in abbreviated form: $x_k = \frac{1}{4}$, for $k = 1, 2, 3, ...$, as k gets larger and larger, and approach inifinity, the terms of the sequence get close and closer to zero.
    * We say that $\frac{1}{k}$ tends to zero as $k$ tends to inifity
    * Or: as k tends to infinity, the limit of the sequence is zero. We write this as:
        * $\displaystyle{\lim_{k \to \infty}} \frac{1}{k} = 0$
    * When a sequence posseses a limit, it is said to **converge**.
    * However, not all sequences have a limit.
        * The sequence defined by $x_k = 3k - 2$, which is $1, 4, 7, 10, ...$ is one example.
    * As $k$ gets larger and larger, so do the terms of the sequence.
        * We say these sequences **diverge**.
    * Example: Write down the first four terms of the sequence $x_k = 3 + \frac{1}{k^2}$, $k = 1, 2, 3, ...$
        * $3 + \frac{1}{1^2} = 4$
        * $3 + \frac{1}{2^2} = 3 \frac{1}{4}$
        * $3 + \frac{1}{2^3} = 3 \frac{1}{8}$
        * $3 + \frac{1}{2^4} = 3 \frac{1}{16}$
        * As more terms are included we see that $x_k$ approaches 3 because the quantity $\frac{1}{k^7}$ becomes smaller and smaller.
            * We can write: $\displaystyle{\lim_{k \to \infty}} ( 3 + \frac{1}{k^2} ) = 3$
                * ie the sequence converges to 3.
    * What is the limit of a convergent sequence?
        * The limit of a convergent sequence, refers to value of the sequence as the kth term approach \infty.
    * Find the limit of each of the following sequences, as $k$ tends to infinity.
        * a) $x_k = k, k = 1, 2, 3 ...$
            * $\displaystyle{\lim_{k \to \infty} (k) = \infty}$ (no limit)
        * b) $x_k = k^2, k = 1, 2, 3 ...$
            * $\displaystyle{\lim_{k \to \infty} (k^2) = \infty}$ (no limit)
        * c) $x_k = \frac{100}{k}, k = 1, 2, 3$
            * $\displaystyle{\lim_{k \to \infty}} (\frac{100}{k}) = 0$
        * d) $x_k = \frac{1}{k^2}, k = 1, 2, 3, ...$
            * $\displaystyle{\lim_{k \to \infty}} (\frac{1}{k^2}) = 0$
        * e) $x_k = k + 1$, $k = , 1, 2, 3, ...$
            * $\displaystyle{\lim_{k \to \infty}} (k + 1) = \infty$ (no limit)
        * f) $x_k = 2^{k}$, $k = 1, 2, 3, ...$
            * $\displaystyle{\lim_{k \to \infty}} (2^{k}) = \infty$ (no limit)
        * g) $x_k = (\frac{1}{2})^{k}, k= 1, 2, 3, ....$
            * $\displaystyle{\lim_{k \to \infty}} (\frac{1}{2})^k = 0$
        * h) $x_k = 7 + \frac{3}{k^2}, k = 1, 2, 3,...$
            * $\displaystyle{\lim_{k \to \infty}} (7 + \frac{3}{k^2}) = 7$
        * i) $x_k = \frac{k+1}{k}, k = 1, 2, 3, ...$
            * $\displaystyle{\lim_{k \to \infty}} (\frac{k + 1}{k}) = 1$
* [Series and Sigma Notation](Series%20and%20Sigma%20Notation)
    * [series](../../../../permanent/series.md)
        * If the terms of a sequence are added, the result is called series.
        * For example, if you add the term: 1, 2, 3, 4, 5 you get: 1 + 2 + 3 + 4 + 5
        * Clearly a series is a sum: if the series contains a finite number of terms, we are able to add them all up and obtain the sum of the series.
        * If the series contains an infinite nunmber of terms, teh situation is more complicated.
        * An inifinte series may have a finite sum, in wich case iti s said to converge.
        * Alternatively, it may not have, and then it is said to diverge.
    * Sigma notation
        * $\sum$ notation provides a concise way of writing long sums:
            * 1 + 2 + 3 + 4 + 5 + ... + 10 + 11 + 12
            * Can be written: $\sum\limits_{k=1}^{k=12} k$
                * The lowest most value of k is written below.
                * The upper-most on top.
                * Sometimes the $k=$ part can be omitted.
        * Examples:
            * Write out explicitly what is mean by:
                * $\sum\limits_{k=1}^{k=5} k^3$
                * $1^3 + 2^3 + 3^3 + 4^3 + 5^3 = 1 + 8 + 27 + 64 + 125 = 225$
            * Express $\frac{1}{1} + \frac{1}{2} + \frac{1}{3} + \frac{1}{4}$ using Sigma notation.
                * $\sum\limits_{k=1}^{k=4} \frac{1}{k}$
            * Write the sum $x_1 + x_2 + x_3 + x_4 + ... + x_{19} + x_{20}$
                * The sum may be written as
                    * $\sum\limits_{k=1}^{k=20} x_k$
        * It does not have to be the letter $k$, any letter can be used.
        * There's also a little trick to alternate the signs of numbers betwee n+ and -. A factor of (-1)^{k} mean the terms in the series alternate in sign.
        * Examples:
            * Write out fully what is mean by $\sum\limits_{k=1}^{4} (-1)^{k}2^{k}$
                * $-1^{1}2^{1} = -2$
                * $-1^{2}2^{2} = 4$
                * $-1^{3}2^{3} = -8$
                * $-1^{4}2^{4} = 16$
            * Write out fully what is meant by $\sum\limits_{i=0}^{5} \frac{(-1)^{i + 1}}{2i + 1}$
                * i=0 = -1^1 / (2 *0 + 1) = -1/1
                * $(-1)^2 / (2*1 + 1)$ = 1/3
                * $(-1)^3 / (2*2 + 1)$ = -1 / 5
                * $(-1)^4 / (2*3 + 1)$ = 1 / 7
                * $(-1)^5 / (2*4 + 1)$ = -1 / 9
                * $(-1)^6 / (2*5 + 1)$ = 1 / 11
                * Result: $-1 + \frac{1}{3} + -(\frac{1}{5}) + \frac{1}{7} + -(\frac{1}{11})$
            * Exercise 12.5
                * 1. Write out fully what is mean by:
                    * a) $\sum\limits_{i=1}^{i=5} i^{2}$
                        * $1^2 + 2^2 + 3^2 + 4^2 + 5^2$
                    * b) $\sum\limits_{k=1}^{4} (2k + 1)^{2}$
                        * $(2+1)^2 + (2 \times 2 + 1)^{2} + (2 \times 3 + 1)^{2} + (2 \times 4 + 1)^{2}$
                        * $3^2 + 5^2 + 7^2 + 9^2$
                    * c) $\sum\limits_{k=0}^{4} (2k + 1)^2$
                        * $(2 \times 0+1)^2 + (2 \times 1 + 1)^2 + (2 \times 2 + 1)^2 + (2 \times 3 +1)^2 + (2 \times 4 + 1)^2$
                        * $1 + 3^2 + 5^2 + 7^2 + 9^2$
                * 2. Write out fully what is meant by:
                    * a) $\sum\limits_{i=1}^{4} \frac{i}{i+1}$
                        * $\frac{1}{2} + \frac{2}{3} + \frac{3}{4} + \frac{4}{5}$
                    * b) $\sum\limits_{n=0}^{3} \frac{n+1}{n+2}$
                        * $\frac{1}{2} + \frac{2}{3} + \frac{3}{4} + \frac{4}{5}$
                * 3. Write out fully what is meant by:
                    * $\sum\limits_{k=1}^{3} \frac{(-1)^k}{k}$
                    * $\frac{-1^{1}}{1} + \frac{(-1)^2}{2} + \frac{(-1)^3}{3} = -1 + \frac{1}{2} + \frac{-1}{3}$
* [Arithmetic Series](Arithmetic%20Series)
    * When we add the terms of an [Arithmetic Sequence](Arithmetic%20Sequence), it's called an Arithmetic Series.
    * There is a formula we can use to find the sum of an arithmetic series:
        * The sum of the first $n$ terms of an arithmetic series with first term $a$ and common difference $d$ is denoted by $S_n$ and given by:
            * $S_n = \frac{n}{2} (2a + (n-1)d)$
    * Examples:
        * Find the sum of the first 10 terms of the aritmetic series with first term 3 and common difference 4.
            * $S_n = \frac{n}{2} (2a + (n-1)d), a=3, d=4$
            * $S_{10} = \frac{10}{2} (2 \times 3 + (10-1) \times 4$
            * $S_{10} = \frac{10}{2} (6 + 9 \times 4)$
            * $S_{10} = \frac{5}{1} (6 + 36)$
            * $S_{10} = \frac{5}{1} (42)$
            * $S_{10} = 5 \times 42 = 210$
            * Test: $3 + 7 + 11 + 15 + 19 + 23 + 27 + 31 + 35 + 39 = 210$
    * Assessment:
        * 1. Explain what is meant by arithmetic series:
            * The sum of the terms of an arithmetic sequence.
        * 2. Write down the formula for the sum of the first $n$ terms of an arithmetic series.
            * $S_n = \frac{n}{2} (2a + (n-1)d)$
    * Exercise 12.6
        * 1. Find the sum of the first 12 terms of the arithmetic series with first term 10 and common difference 8.
            * $S_{12} = \frac{12}{2} (2 \times 10 + (12-1) \times 8$
            * $S_{12} = 6 (20 + 11 \times 8)$
            * $S_{12} = 6 (108) = 648$
        * 2. Find the sum of the first seven terms of the arithmetic series with first term -3 and common difference -2.
            * $S_7 = \frac{7}{2} (2 \times (-3) + (7-1) \times (-2)$
            * $S_7 = \frac{7}{2} (-6 + 6 \times (-2)$
            * $S_7 = \frac{7}{2} 0 = 0$
        * 3. The sum of the arithmetric series is 270. The common difference is 1 and the first term is 4. Calculate the number of terms in the series.
            * $270 = \frac{n}{2} (2(4) + (n-1) \times (1)$
            * $270 = \frac{n}{2} (8 + (n-1))$
            * $270 = \frac{n}{2} (7 + n)$
            * $540 = n \times (7 + n)$ -- multiple both sides by 2.
            * $540 = 7n + n^2$ -- factorise
            * $0 = n^2 + 7n - 540$ --quadratic form where a=1, b=7, c=-540
            * Need 2 numbers that multiple to -540 and add to 7.
                * 27 and -20 work.
                * Can factorise as (n - 20)(n + 27) = 0
                * Setting each factor to 0 give:
                    * n = 20 or n = -27
                * Since a sequence value is positive, we are left with $n=20$.
        * 4. The sum of the first 15 terms of an arithmetic series is 165. Common diff is 2. What's the first term?
            * $165 = \frac{15}{2} (2a + (14) \times 2) = \frac{15}{2} (2a + 28)$
            * 165 = 15a + 210
            * 165 - 210 = 15a
            * -45 = 15a
            * $a = -3$
        * 5. The sum of first 13 terms of an arithmetic series is 0. The first term is 3. Calculate the common difference.
            * $0 = \frac{13}{2} (6 + 12d)$
            * $0 = 39 + 78d$
            * $0 - 39 = 78d$
            * $-39/78 = d$
            * $d = -(\frac{1}{2})$
* [Geometric Series](Geometric%20Series)
    * When terms of a [Geometric Sequence](Geometric%20Sequence) are added.
    * The formula for a geometric series is: $S_n = \frac{a(1-r^n)}{1-r}$
        * Where $n$ = term of sequence, $a$ = first term and $r$ = common ratio (although cannot be equal to 1).
    * Example: Use formula to find S_n, where n=5, a =2 and r = 3:
        * $S_5 = \frac{2(1-r^5)}{1-r}$
        * $S_5 = \frac{2(1 - 243)}{-2}$
        * $S_5 = \frac{2 - 486}{-2}$
        * $S_5 = \frac{-484}{-2}$
        * $S_5 = 242$
* [Infinite Geometric Series](Infinite%20Geometric%20Series)
    * When the terms of an infinite sequence are added we obtain an infinite series.
        * This only works when the sum is finite.
    * The case where a geometric series with a common ratio between -1 and 1:
        * $S_{\infty} = \frac{a}{1-r}$ provided -1 < r < 1
        * If the common ratio is better than 1 or less than -1, then the series does not converge and the sum of an infinite geometric series cannot be found.
    * Examples:
        * Find the sum of an infinite geometric series with first term 2 and common ratio $\frac{1}{3}$
            * $S_{\infty} = \frac{2}{1-\frac{1}{3}} = \frac{2}{\frac{2}{3}} = 3$
            * Again, this only works because r lies between -1 and 1.
