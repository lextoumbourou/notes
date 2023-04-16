---
title: Discrete Mathematics with Applications - Chapter 12
date: 2022-10-11 00:00
category: reference/books
status: draft
---

## Chapter 12. Boolean Algebra and Combinatorial Circuits

* 1854: [George Boole](George Boole)'s book [An Investigation of the Laws of Thought](An Investigation of the Laws of Thought)
    * Was the foundation for Symoblic Logic and Boolean Algebra.
    * Didn't have much application until...
* 1938: [Claude E. Shannon](Claude E. Shannon) used boolean algebra to analyze electrical circuits.
* [Boolean Algebra](../../../permanent/boolean-algebra.md)
    * Mathematical system
    * *"it consists of a nonempty set S with one or more operations defined on S, and a set of axioms that the elements of S satisfy."*
    * Think of it like a skeleton. Though people's exterior differs, under it all we are just a walking skeleton.
    * Likewise, math systems share common properties.
        * Eg Real numbers are an important number system.
* Example
    * Let $U$ be an abritrary set and $P(U)$ its power set.
    * Let $A$, $B$, and $C$ be any three elements of $P(U)$.
    * Recall that union, intersection and complementation operations on $P(U)$ satisfy:
        * Commutative properties
            * $A \cup B = B \cup A$
        * Associative properties
            * $A \cup (B \cup C) = (A \cup B) \cup C$
        * Distributive properties
            * $A \cap (B \cup C) = (A \cap B) \cup (A \cap B)$
            * $A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$
        * Identity properties
            * $A \cup \emptyset = A$
            * $A \cap U = A$
        * Complement properties
            * $A \cup A' = U$
            * $A \cap A' = \emptyset$

Example 12.2

This examples requires understanding 2 functions that I'm currently not familiar with: `lcm` and `gcd`.

Boolean Algebra

A boolean algebra consists of a nonempty set B containing two distinct element 0 and 1, two binary operator + and ., and a unary operator '.

It must satisfy all these conditions for x, y and z in B.

Commutative laws

| +   | .   |
| --- | --- |
|  $x + y = y + x$   |  $x . y = y . x$   |

Associative laws

| +   | .   |
| --- | --- |
|  x + (y + z) = (x + y) + z   |  x . (y . z) = (x . y) . z   |

Distributive laws

| +                               | .   |
| ------------------------------- | --- |
| x . (y + z) = (x . y) + (x . z) | x + (y . z) = (x + y) . (x + z)  |

Identity laws

| +         | .         |
| --------- | --------- |
| x + 0 = x | x . 1 = x |

Complement laws

| +   | .   |
| --- | --- |
|  x + x' = 1   | x . x' = 0    |

* The operations $+$, $.$, and $'$ are called *sum*, *product* and *complementation* respectively.
    * They are generic operators: they do no stand for addition or multiplication.
* The elements 0 and 1 are the zero element and the unit element.
    * They are generic symbols: they do not need to represent 0 and 1.
    * In the set example, the zero element is $\emptyset$ and unit element is $U$.
* The operator . in x . y is often omitted: $xy$
* No parenthese need be used when there won't be confusion:
    * $(xy) + (xz) = xy + xz$
    * $x + y + z = x + ( y + z) = (x + y) + z$
* Order is important:
    * Parenthesizes subexpressions are evaluated.
    * Complementation (NOT), followed by Product (AND), followed by Sum (OR)
        * $xy + zx' = (xy) + [z(x')]$
    * The 10 axioms are paired off in two columns.
        * In each pair, an axiom can be obtained from the other by swapping $+$ with $.$ and $0$ with $1$.
        * These are dual axioms. For instance, the dual of x + x' = 1 (axiom 9) is x . x' = 0 (axiom 10).
            * The dual of every statement is true. This is called **principle of duality**.
