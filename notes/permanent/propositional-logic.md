---
title: Propositional Logic
date: 2023-10-12 00:00
modified: 2023-10-12 00:00
cover: /_media/prop-logic-cover.png
summary: A system that deals with statements that are either true or false
tags:
    - DiscreteMath
---

A system that deals with [Proposition](proposition.md) (statements).

## Proposition

A declarative sentence that is either true or false (but not both) is a proposition.

Also known as a statement.

These sentences would be considered propositions:

* It is Thursday today (true).
* I am 14 years old (false).
* 1 + 1 = 3 (false).

Usually denoted using lowercase letters $p$, $q$, $r$, $s$ or $t$.

p = It rained yesterday.
q = I am happy.

The truthfulness or falsity of a proposition is called its [Truth Value](Truth%20Value). Denoted by $T$ or $F$, or 1 and 0 in computer science.

We can use connectives to change or combine the meaning of propositions. For example, $\neg p$ negates the value of p. If it's true, it becomes false and vice versa.

## [Truth Table](../journal/permanent/truth-table.md)

A truth table allows us to consider all possible combinations of proposition logic systems.

For example, consider $\neg p$:

| $p$ | $\neg p$ |
| --- | -------- |
| 1   | 0        |
| 0   | 1         |

We can use truth tables to help us understand the truth values of other connectives within propositional logic.
## Connectives

### [Negation](logical-negation.md) (NOT)

Symbol: $\neg p$

An operator that negates a proposition.

* $p$ = I will pass my exam.
* $\neg \ p$ = I will NOT pass my exam.

In [Boolean Algebra](Boolean%20Algebra), it's equivalent to $1 - T(p)$

Truth table

| $p$ | $\neg p$ |
| --- | -------- |
| 1   | 0        |
| 0   | 1         |

## [Disjunction](logical-disjunction.md) (OR)

Symbol: $p \lor q$

True when p OR q is true.

Truth table

 | $p$ | $q$ | $p \lor q$ |
 | --- | --- | ----------- |
 | 1   | 0   | 1           |
 | 0   | 1   | 1           |
 | 1   | 1   | 1           |
 | 0   | 0   | 0           |

Equivalent to $\max(T(p), T(q))$

## [Conjunction](conjunction.md) (AND)

Symbol: $p \land q$

True only when p AND q is true.

Truth table:

| $p$ | $q$ | $p \land q$ |
| --- | --- | ----------- |
| 1   | 0   | 0           |
| 0   | 1   | 0           |
| 1   | 1   | 1           |
| 0   | 0   | 0           |

Equivalent to multiplication $T(p) \times T(q)$

## [Implication](logical-implication.md) (If...Then)

Symbol: $p \rightarrow q$

If p is true, then q is true.

Truth table

| $p$ | $q$ | $p \rightarrow q$ |
| --- | --- | ----------------- |
| 1   | 1   | 1                 |
| 0   | 1   | 1                 |
| 1   | 0   | 0                 |
| 0   | 0   | 1                 |

Think of it as a promise. Only false when promise is broken.

## [Bi-conditional](logical-biconditional.md): $\leftrightarrow$

Symbol: $p \leftrightarrow q$

Truth table

| $p$ | $q$ | $p \leftrightarrow q$ |
| --- | --- | ----------------- |
| 1   | 1   | 1                 |
| 0   | 1   | 0                 |
| 1   | 0   | 0                 |
| 0   | 0   | 1                 |

Equivalent to equality check: $T(p) = T(q)$

## [Exclusive-Or](logical-xor.md)

Symbol: $p \oplus q$

p or q but not both.

Also called XOR.

Truth Table

| $p$ | $q$ | $p \oplus q$ |
| --- | --- | ----------------- |
| 1   | 1   | 0                 |
| 0   | 1   | 1                 |
| 1   | 0   | 1                 |
| 0   | 0   | 0                 |

Truth table is opposite of bi-conditional.
## Operator Precendence

1. $\neg$
2. $\land$
3. $\lor$
4. $\rightarrow$
5. $\leftrightarrow$
