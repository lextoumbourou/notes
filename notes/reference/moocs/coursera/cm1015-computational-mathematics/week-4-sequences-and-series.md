---
title: Week 4 - Sequences and series
date: 2023-05-10 00:00
status: draft
---


* [[Series]]
    * A specific type of sequence.
        * Notation for sequences: $a_n$ with $n=0,1,2,...$
    * A sequence that is derived from another sequence.
    * Consider generic sequence $\{a_n\}$ with $n=0,1,2$
        *  We can look at sum of elements:
            * $a_0, \ a_0+a_1, \ a_0+a_1+a_2, \ a_0 + a_1 +a_2+a_3,...$
        * These sums define another sequence ${s_n} n=0,1,2,...$ with $s_n = a_0 + a_1 + ... a_n \rightarrow s_0 = a_0, s_1 = a_0 + a+1, s_2 = a_0 + a_1 + a_2, ...$
        * ${s_n}$ is called a series, it is a type of sequence that is obtained as sum of the elements of another sequence
        * A series is also indicated with $\sum \rightarrow s_n = \sum^{n}_{i=0} a_i$
    * Example: given the arithmetic sequence $a_n = a_0 + n \times q$ can we find an explicit expression for $s_n  = \sum^{n}_{k=0} a_k$.
        * First need to look at [Mathematical Induction](../../../../permanent/induction.md)
            * If a property is valid for $n=0$, and if, assuming it is valid for $n$, it remains valid also for $n + 1$, then the property is valid for any $n$.
            * Prove using Mathematical Induction, that for the arithmetic sequence, the relative series $s_n =  \sum^{n}_{k=0} (a_0 + k \times x) = \color{blue}{(n+1)(2 a_0 + n \times q) / 2 \ \ \ (*)}$ 
                * First we should prove that the $\color{blue}{\text{blue}}$ right-hand expression is true for $n=0$.
                * Then prove that $s_{n+1} = s_n + a_{n+1} = \color{blue} {\text{(*)}} \color{black}{ + a_0 + (n+1) \times q}$ 
                * $\color{blue}{(n+1)(2 a_0 + n \times q) / 2} \color{black}{+ a_0 + (n+1) \times q} = \color{blue}{(n + 1 + 1)(2a_0 + (n+1) \times q) / 2}$
        * Another example, geometric sequence:
            * $s_n = \sum_{k=0}^{n} (a_0 \times q^{k}) = \color{blue} a_0 \ \frac{1-q^{n+1}}{1-q}$
            * Hint: prove $\color{blue}{\text{blue side}}$ is true for $n=0$, then prove that $s_{n+1} = s_n + a_{n+1} = \color{blue}{(*)} \color{black}{+ a_0 \times q^{n+1}}$ gives again $\color{blue}{(*)}$ with $n \rightarrow n + 1$
* [Mathematical Induction](../../../../permanent/induction.md) example
    * $s(n) = 0 + 1 + 2 + ... + n$
    * This formula gives the correct value: $s(n) = n(n+1)/2$
    * Proof
        * First prove true if n = 0
            * $s(0) = 0$
        * If for $n = k$ and true for $n = k + 1$, then true for any $n$.
            * If we assume $s(k) = k(k + 1) / 2$
            * $s(k+1) = 1 + 2 + .... + k + k + 1$
            * Can rewrite as $s(k) + s(k + 1)$
            * Can rewrite as $\frac{k(k+1)}{2} + {k+1}$
            * Can factor out to $(k+1)(\frac{k}{2} +1)$
            * If you take the lowest common multiple in 2nd factor: $\frac{(k+1)(k+2)}{2}$
            * Note it's just equal to above formula when $s(k+1)$: $\frac{(k+1)(k+2)}{2}$
* Introduction to modular arithmetic
    * Think of it as a way to "classify integers", an arithmetic over integers.
    * Originally formulated by mathematician and physicist Carl F. Gauss, a system of arithmetic for integers.
    * Numerous applications from number theory to library and bank classifications sysmtes to cryptography.
        * Basic notion congruence between integers
        * Two numbers a and b are congruent "mod 2" if they have the same remainder when divided by 2.
        * Congruent symbol: $\equiv$
        * In general, we say $a \equiv b$ (mod k) $\Leftrightarrow a = nk + R, b=mk + R$ 
            * Or in other words, if you divide by n you get the same remainder.
    * Clock arithmetic
        * if it is 8AM after 7 hours it will be 15 or 3 PM as it is "mod 12".
            * $\rightarrow 15 \equiv 3$ $(\mod 12)$
        * since $15/12=1$ with $R=3$ $3/12=0$ with $R=3$
    * Mod k is use to map by congruence all integers to the subset of non-negative integers smaller than k that is Min_k = {0, 1, 2, ... k-1}
    * So in mod 12 we can map, by congruence, all integers to one of the integers in Min_12 = {0, 1, 2, 3, 4,5,6,7,8,9,10,11}
    * With negative integers when we divide by k we need to get a non-negative remainder.
        * Ex: -12 mod 12
        * You cannot do -12/12 = -1 with R = -5 negative (wrong) but -12/12 = -2 with R=7 positive
            * $\rightarrow -17 \equiv 7$ (mod 12)

### Essential Reading
            
* Theory of Divisibility
    * Basic concepts and properties of divisibility
        * Let $a$ and $b$ be integers with $a \ne  0$
            * We say a divides b, denoted by $a \mid b$, if there's an integer $c$ such that $b = ac$
                * In this case, $a$ is a *divisor* 
                * $b$ is a *multiple* of $a$.
            * When a divides b, we say that a is a divisor (or factor) of b, and b is a multiple of a.
            * If a does not divide b, we write $x\nmid y$
            * If $a \mid b$ and $0 < a < b$, then a is called a proper divisor of $b$.
        * We never use 0 as the left member of the pair of integers in a | b, however 0 may occur as the right member of the pair, thus a | 0 for every integer a not zero.
        * Example: the integer 200 has the following positive divisors):
            * 1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200:
                * 8 \mid 200
                * 50 \mid 200
                * 36 \nmid 200
        * [[Trivial Divisor]]
            * A trivial divisor of $n$ is $1$ or $n$ itself.