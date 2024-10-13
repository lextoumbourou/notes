---
title: Week 4 - Sequences and Series
date: 2023-05-10 00:00
status: draft
---

* [[../../../../permanent/series]]
    * A specific type of sequence.
        * Notation for sequences: $a_n$ with $n=0,1,2,...$
    * A sequence that is derived from another sequence.
    * Consider generic sequence $\{a_n\}$ with $n=0,1,2$
        * We can look at sum of elements:
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
    * Proof:
        * First prove true if $n = 0$
            * $s(0) = 0$
        * If for $n = k$ and true for $n = k + 1$, then true for any $n$.
            * If we assume $s(k) = k(k + 1) / 2$
            * $s(k+1) = 1 + 2 + .... + k + k + 1$
            * Can rewrite as $s(k) + s(k + 1)$
            * Can rewrite as $\frac{k(k+1)}{2} + {k+1}$
            * Can factor out to $(k+1)(\frac{k}{2} +1)$
            * If you take the lowest common multiple in 2nd factor: $\frac{(k+1)(k+2)}{2}$
            * Note it's just equal to above formula when $s(k+1)$: $\frac{(k+1)(k+2)}{2}$
