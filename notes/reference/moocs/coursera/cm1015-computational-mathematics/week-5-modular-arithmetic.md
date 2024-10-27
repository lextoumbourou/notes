---
title: Week 5 - Modular Arithmetic
date: 2023-05-22 00:00
modified: 2023-05-22 00:00
status: draft
---

* [[Modular Arithmetic]]
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
                * $8 \mid 200$
                * $50 \mid 200$
                * $36 \nmid 200$
        * A divisor of $n$ is called a **trivial divisor of n** if it is either 1 or $n$ itself.
        * [[Trivial Divisor]]
            * A trivial divisor of $n$ is $1$ or $n$ itself.
            * A non-trivial divisor is a divisor of $n$ that is neither 1 or $n$.
            * Example:
                * For the integer 18, 1 and 18 are the trivial divisors, whereas 2, 3, 6 and 9 are the nontrivial divisors.
                * The integer 191 has only 2 trivial divisors and does not have any nontrivial divisors.
                * Basic properties of divisibility:
                    * Let $a$, $b$ and $c$ be integers. Then:
                        * 1) if $a \mid b$ and $a \mid c$ then $a \mid (b + c)$
                        * 2) if $a \mid b$ and $a \mid bc$ for any integer $c$.
                        * 3) if $a \mid b$ and $b \mid c$ then $a \mid c$
                    * Proof:
                        * 1) Since $a \mid b$ and $a \mid c$, we have:
                            * $b = ma$, $c = na$, $m, n \in \mathbb{Z}$
                            * Then $b + c = (m + n)a$ Hence, $a \mid (m + n)a$ since $m + n$ is an integer.
                        * 2) Since $a \mid b$ we have: $b = ma, m \in \mathbb{Z}$
                            * Multiplying both sides of this equality by $c$ gives: $bc = (mc)a$
                            * which gives $a | bc$ for all integers $c$ (whether or not c = 0).
                        * 3) Since $a \mid b$ and $b \mid c$ there exists integers $m$ and $n$ such that $b = ma$, $c = nb$.
                            * Thus, $c = (mn)a$ Since $mn$ is an integer the result follows.
        * Exercise 1.2.1. Let $a, b$ and $c$ be integers. Show that:
            * 1) $1 \mid a$, $a \mid a$, $a \mid 0$
                * 1 is a trivial divisor of every integer.
                * a is a trivial divisor of integer a.
                * a | 0 for every integer a.
            * 2) if $a \mid b$ and $b \mid a$ then $a = \pm b$
            * 3) if $a \mid b$ and $a \mid c$, then for all integers $m$ and $n$ we have $a \mid (mb + nc)$
            * 4) if $a \mid b$ and $a$ and $b$ are positive integers, then $a < b$
