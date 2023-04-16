---
title: Week 18 - Relations B
date: 2022-03-01 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
modified: 2023-04-08 00:00
---

## 9.201 Equivalence relations and equivalence classes

* Definition of [Equivalence Relation](permanent/equivalence-relation.md)
    * Let R be a relation of elements on a set S. R is an equivalence relation if and only if R is reflexive, symmetric and transitive.
* Example 1
    * Let R be relation of elements in Z:
        * $R = \{ (a, b) \in Z^2 | a \mod 2 = b \mod 2 \}$
    * We have already proved that this relation is:
        * Reflexive as a $R a, \in a \forall Z$
        * Symmetric as if a R b then b R a, \forall a, b \in Z
        * Transitive as if a R b and b R c then a R c, \forall a, b, c \in Z
        * This is an equivalence relationship.
* Example 2
    * Let R be a relation of elements in Z:
        * $R = \{ (a, b) \in Z^2 | a \leq b \}$
        * Already proved relationship is :
            * Reflex as a R a for all a in Z
            * Transitive as if $a \ R \ b$ and $b \ R \ c$ then $a \ R \ c$, $\forall a, b, c \in Z$
            * Not symmetric as $2 \leq 3$ but $3 \not \lt 2, \forall a, b \in Z$
        * This is not equivalence.
* Definition of [Equivalence Class](permanent/equivalence-class.md)
    * Let R be an equivalence relation on a set S. Then, the equivalence class of a \in S is:
        * a subset of S containing all the elements related to a through R.
        * $|a| = \{x: x \in S \text { and } x \ R \ a\}$
* Example 1
    * Let $S = \{1, 2, 3, 4\}$ and R be a relation on elements in S:
        * $R = \{ (a, b) \in S^2 | a \mod 2 = b \mod 2 \}$
    * R is an equivalence relation with 2 equivalence classes:
        * `[1] = [3] = {1, 3}
        * `[2] = [4] == {2, 4}`
* Example 2
    * Let $Z = \{1, 2, 3, 4, 5\}$ and R be relation of elements in Z:
        * $R = \{ (a, b) \in Z^2 | a - b \text{ is an even number } \}$
    * R = { (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (2, 4), (4, 2), (1, 3), (3, 1), (1, 5), (5, 1), (3, 5), (5, 3) }
    * R is an equivalence relation with 2 equivalence classe:
        * `[1] = [3] = [5] = {1, 3, 5}`
        * `[2] = [4] = {2, 4}`

### 9.203 Partial and total order

* Definition of a [Partial Order](permanent/partial-order.md).
    * Let $R$ be a relation on elements in set $S$. $R$ is a partial order if and only if $R$ is:
        * reflexive
        * anti-symmetric
        * transitive.
    * Example 1
        * Let R be a relation of elements in Z:
            * $R = \{ (a, b) \in Z^2 | a \leq b \}$
        * It can easily be proved that R is:
            * reflexive as $a \leq a$, $\forall a \in Z$
            * transitive as if $a \leq b$ and $b \leq c$ then $a \leq c$, $\forall a, b \in Z$
            * anti-symmetric as if $a \leq b$ and $b \leq a$ then $a = b$, $\forall a, b \in Z$
        * Therefore, R is a partial order.
    * Example 2
        * Let $R$ be a relation of elements in Z+:
            * $R = \{ (a, b) \in Z+ | a \text{ divides } b \}$
        * It can easily be proven that R is:
            * reflexitive as a divides $a, \forall a \in Z+$
            * transitive as if $a \text{ divides } b$ and $b \text{ divides } c$ then $a \text{ divides } c$, $\forall a, b ,c \in Z+$
            * anti-symmetric as if $a \text{ divides } b$ and $b \text{ divides } a$ then $a = b$, $\forall a, b \in Z+$
        * Therefore, R is a partial order.
* Definition of a [Total Order](permanent/total-order.md)
    * Let R be a relation on elements in a set S.
    * R is a total order if and only if:
        * R is a partial order
        * $\forall a,b \in S$ we have either $a \ R \ b$ or $b  \ R \ a$.
    * Example 1
        * Let R be a relation of elements in Z:
            * $R = \{ (a, b) \in Z^2 | a \leq b \}$
        * It has been shown that R is a partial order.
        * Also, $\forall a, b \in Z$, $a \leq b$ or $b \leq a$ is true.
    * Example 2
        * Let R be a relation on elements in Z+:
            * $R = \{ (a, b) \in Z+ | a \text{ divides } b \}$
        * We proved R is a partial order
        * However, Z* contains elements that are incomparable, such as 5 and 7.
        * R is not totally ordered.

# Peer-graded Assignment: 9.207 Relations

## Question

Let R be a relation on $Z$ given by $x \ R \ y$ if and only if $x^2 - y^2$ is divisible by 3. Show that this relation is an equivalence relation and find its corresponding equivalence classes.

## Answer

To prove $x \ R \ y$ has an equivalence relationship, we must show it's *reflexive*, *symmetric* and *transitive*.

We know it's reflexive, because for any integer $x^2 - x^2 = 0$, which is divisible by 3.

We know it's symmetric, because $y^2 - x^2 = -(x^2 - y^2)$. If $x^2 - y^2$ is divisible by three, $-(x^2-y^2)$ is also divisible by three.

When know it's transitive, where $x \ R \ y$ and $y \ R \ z$ therefore $x \ R \ y$, because we know $(x^2 - y^2) + (y^2 - z^2) = x^2 - z^2$, so when $x^2 - y^2$ and $y^2 - z^2$ are both divisible by 3, $x^2 - z^2$ will also be divisible by 3.

---

To find the corresponding equivalence classes, we must find all integers related to a given $x$.

Suppose $y$ is related to $x$, then $x^2 - y^3$ is divisible by 3. That means that all means they have the same residual mod 3.

There are 3 residual mods possible: 0, 1, 2.

Therefore we can assign the integers into these equivalence classes:

* $0 = \{..., -6, -3, 0, 3, 6, ...\}$
* $1 = \{..., -5, -2, 1, 4, 7, ...\}$
* $2 = \{..., -4, -1, 2, 5, 8, ...\}$

However, 1 and 2 are the same. So it's only 2 equivalence classes.

# Peer Review

## Question 1

**1. Reflexive**

$R$ is reflexive if $\forall x \in S$, $x \ R \ x$

Example: Equality is reflexive, but < is not reflexive.

**2. Symmetric**

$R$ is symmetric if $\forall x, y \in S$, if $x \ R \ y$ then $y \ R\ x$.

Example: a marriage is symmetric, but *is parent of is not* symmetric.

**3. Anti-symmetric**

$R$ is anti-symmetric if $\forall x, y \in S$, if $(x \ R \ y$ and $y \ R \ x)$ then x = y

Example:
* subset ($\subseteq$) is antisymmetric relation. If x is a subset of y, and y is a subset of x, then by definition x = y.
* an ancestor. If x is an ancestor or y, and y is an ancestor of y, they must be the same person.

Counterexample:
* marriage from above. a husband is related to his wife, the wife is related to the husband, but they are not the same.

**4. Transitive**

$R$ is transitive if $\forall x, y, z \in S$, if $x\ R \ y$ and $y \ R \ z$ then $x \ R \ z$

Example: is in immediate family is transitive, is a parent is not.

**5. An equivalence relation.**

$R$ is an equivalence relation if it is reflexive, symmetric and transitive.

Example:
* is equal to on the set of real number is an equivalence relation.
* is less or equal ($\leq$) is not equivalence relation.

**6. A partial order.**

$R$ is a partial order if it is reflexive, antisymmetric and transitive

Example:

The relationship divides on the set of positive integers (natural numbers $\mathbf{N}$), is reflexive, antisymmetric and transitive. But the same relation applied to the set of integers $\mathbf{Z}$ is not antisymmetric since $-2 | 2$ and $2 | -2$ but $-2 \neq 2$

## Question 2

Let $S = \{a, b, c\}$ and $A = \{(c, c), (a, b), (b, b), (b, c), (c, b)\}$

Define a relation $R$ on $S$ by "$x$ is related to $y$ whenever $(x, y) \in A$"

1. Draw the relationship digraph
2. The relationship $R$ is not reflexive. What pair (x, y) should be added to $A$ to make $R$ refexive?

(a, a)

**Correct**

3. The relationship $R$ is not symmetric. What pair (x, y) should be added to A to make R symmetric?

(b, a)

**Correct**

4. The relation R is not anti-symmetric. What pair (x, y) should be **removed** to make R anti-symmetric?

(b, c) or (c, b)

Right now b R c and c R b but b != c

**Correct**

5. The relation R is not transitive. What pair (x, y) should be added to A to make R transitive?

(a, c)

**Correct**

## Question 3

The following relations are defined on a set S = {a, b, c}

$R_1$ is the relation given by $\{(a, a), (a, b), (a, c), (b, a), (b, b), (c, a), (c, c)\}$
$R_2$ is given by $\{(a, a), (a, b), (b, a), (b, b), (c, c)\}$
$R_3$ is given by $\{(a, b), (a, c), (b, a), (b, c), (c, a), (c, b)\}$
$R_4$ is given by $\{(a, a), (a, b), (a, c), (b,b), (b, c), (c, c)\}$

|     | Reflexive | Symmetric | Antisymmetric | Transitive | Equivalence Relation | Partial Order |
| --- | --------- | --------- | ------------- | ---------- | -------------------- | ------------- |
| R_1 | Y         | Y         | N             | N          | N                    | N             |
| R_2 | Y         | Y         | N             | **Y**      | Y                    | N             |
| R_3 | N         | Y         | N             | N          | Y                    | N             |
| R_4 | Y         | N         | Y             | Y          | N                    | Y             |

Note, that $R_2$ is considered transitive. We have (a, b) and (b, a) therefore we need (a, a) which we have.
$R_3$ is not considered transtive. We have (a, b) and (b, a) therefore we need (a, a) to be transitive, which is missing.

$R_2$ is the only equivalence relation.
a related to b
c related to c only.

So there are $T = \{[a], [c]\}$ equivalence classes.

## Question 4

Let $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9\}$ and let P be the partition on S given by = $\{ \{1, 4, 7\}, \{2, 5, 8\}, \{3, 6, 9\}\}$

Define $R$ to be the equivalence relation associated to $P$.

1. Give 2 conditions for $P$ to be a partition.

* Let $P_1 ... P_N$ be the subsets of $P$. P is a partition as each subset , $P_1 \cup P_2 \cup P_3 = P$
* We also know that $P$ is a partition as $P_1 \cap P_2 \cap P_3 = \emptyset$

2. Draw the relationship digraph.

On paper. It's 3 disjointed graphs.

3. Write down the equivalence class [5] as a set.

Apparently it's {2, 5, 8}

So the equivalence class refers to the elements that are in the same class?

## Question 5

Let $S = \mathbb{Z} \ x \ \mathbb{N}^{+}$ and let $\mathbb{R}$ be a relation on S defined as follows:

$(a, b) \ R \ (c, d)$ whenever $ad = bc$

Show that $R$ is an equivalence relation.

* R is reflexive. As $\forall (a, b) \in S$, $(a, b) \ R \ (a, b)$
* R is symmetric as $\forall(a, b), (c, d) \in S$ if $(a, b) \ R \ (c, d)$ then ad = bc AND cb = da then $(c, d) R (a, b)$
* R is transitive as $\forall (a, b), (c, d), (e, f) \in S$. $(a, b) \ R \ (c, d)$ and $(c, d) \ R \ (e, f)$ => ad = bc and cf = de => cbde = dacf => be = af => $(a, b) \ R \ (e, f)$

Define the equivalence class generated by $(a, b)$ for $a \in \mathbb{Z}$ and $b \in \mathbb{N}^{+}$

$[(a, b)] = \{ (m, n): (a, b) \in \mathbb{Z} \ x \ \mathbb{N}^{+} \text{ and } \frac{a}{b} = \frac{m}{n} \}$

## Question 6

Let A and B be two sets where:

$A = \{ \text{ France }, \text{ Germany }, \text{ Switzerland }, \text{ England }, \text{ Morocco } \}$ and

$B = \{ \text{ French }, \text{ German }, \text{ English }, \text{ Arabic } \}$.

Let R be a relation defined from A to B, given by $a \ R \ b$ when b is a national language of a.

The national language of each of the countries is as follows:

* French for France
* German for Germany
* English for England
* Arabic for Morocco
* Switzerland has 2 national languages, French and German.

Find the logical matrix for the relation R.

|             | French | Germany | English | Arabic |
| ----------- | ------ | ------- | ------- | ------ |
| French      | 1      | 0       | 0       | 0      |
| Germany     | 0      | 1       | 0       | 0      |
| England     | 0      | 0       | 1       | 0       |
| Morocco     | 0      | 0       | 0       | 1      |
| Switzerland | 1      | 1       | 0       | 0      |

## Question 7

For each of the following relations on the set of all people, state if it is an equivalence relation. Explain your answer.

1. R_1 = { (x, y) | x and y are the same height }

R_1 is flexive as every $x$ is the same height as itself, or: $x \ R \ x$
R_1 is symmetic as $\forall x, y \in S$ if $x \ R \ y$ then $y \ R \ x$
R_1 is transitive as $\forall x, y, z \in S$ if $x \ R \ y$ and $x \ R \ z$ then $x \ R \ z$

Therefore, this is an *Equivalence Relationship*.

**Correct**

2. R_2 = { (x, y) | x and y have, at some point, lived in the same country. }

R_2 is reflexive as $x \ R \ x$. $x$ lives in the same country as itself.
R_2 is symmetric as $\forall x, y \in S$ if $x \ R \ y$ ie x and y where in the same country, then $y \ R \ x$, as again, they must be in the same country.
R_2 is NOT transitive. If x, y both lived in Australia, and y, z both lived in Malaysia, it does not imply x lived in the same country.

Therefore, it is NOT an Equivalence Relationship.

**Correct**

3. R_3 = { (x, y) | x and y have the same first name }

$R_3$ is reflexive.
$R_3$ is symmetric.
$R_3$ is transitive.

Therefore, $R_3$ is an equivalence relationship.

4. R_4 = { (x, y) | x is taller than y }

R_4 is not reflexive as x is not taller than x.

Therefore, it it not an equivalence relationship.

5. R_5 = { (x, y) | x and y have the same colour hair }

R_5 is reflexive. x R x
R_5 is symmetric. x R y and y R x
R_5 is transitive. x R y and y R z therefore x R y

Therefore, it is an equivalence relationship.

## Question 8

Let S = { {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5}}

X is related to Y whenever $X \subseteq Y$

1. Draw the relationship digraph.

![week-18-problem-sheet-relationship-bigraph](../../../../journal/_media/week-18-problem-sheet-relationship-bigraph.png)

2. Determine whether or not $R$ is reflexive, symmetric, antisymmetric or transitive. Give a brief justification for each of your answers.

* R is reflexive as $x \ R \ x$, for example $\{1\} \subseteq \{1\}$
* R is NOT symmetric. if $x \ R \ y$ then $y \ R \ x$ is NOT true. For example, $\{1\} \subseteq \{1, 2\}, \{2, 1\} \subseteq \{1, 2\}$
* R is anti-symmetric. if $x \ R \ y$ and $y \ R \ x$ then $x = y$.
* R is transitive. If $x \ R \ y$ and $y \ R \ z$, then $x \ R \ z$

3. State, with reasons, whether or not $R$ is an equivalence relation, whether or not it is a partial order and whether or not it is a total order.

R is not an equivalence relation as it is NOT symmetric.
R is a partial order as it is reflexive, anti-symmetric and transitive.
R is a total order, as it is a partial order and every 2 elements are comparable. $\forall a,b \in S$ we have either $a \ R \ b$ or $b  \ R \ a$.

## Question 9

Let $S = {a, b, c, d}$ and let $A \subseteq S \ x \ S$ be given by:

$\{ (a, a), (a, c), (b, b), (b, d), (c, a), (c, c), (d, b), (d, d) \}$

A relation $R$ on $S$ is defined by:

$x$ is related to $y$ whenever $(x, y) \in A$

1. Draw the relationship digraph

![week-18-problem-sheet-q9-digraph](../../../../journal/_media/week-18-problem-sheet-q9-digraph.png)

2. Determine whether or not $R$ is reflexive, symmetric, antisymmetric or transitive. Give brief justification of each answer.

* $R$ is reflexive as $\forall x \in S$, $x \ R \ x$
* $R$ is symmetric as $\forall x, y \in S$, if $x \ R \ y$ then $y \ R \ z$
* $R$ is NOT anti-symmetric as $\forall x, y \in S$, if $x \ R \ y$ and $y \ R \ x$, it DOES NOT imply $x = y$
* $R$ is transitive. $\forall x, y, z \in S$, if $x \ R \ y$ and $y \ R \ z$, then $x \ R \ z$

$R$ is a equivalence relationship as it is reflexive, symettric and transitive.
$R$ is not a partial order as it is NOT anti-symmetric.
$R$ is not a total order as it is NOT a partial order.

## Question 10

Let $R$ be a relation from a set $A$ to a set $B$. The inverse of $R$, denoted $R^{-1}$, is the relation from $B$ to $A$ defined by $R^{-1} = \{ (y, x) : (x, y) \in R \}$

Given a relation $R$ from $A = \{ 2, 3, 4 \}$ to $B = \{ 3, 4, 5, 6, 7 \}$ defined by $(x, y) \in R$ if $x$ divides $y$.

Elements: (2, 4), (2, 6), (3, 3), (3, 6), (4, 4)

$M_r$ of $R$

|     | 3   | 4   | 5   | 6   | 7   |
| --- | --- | --- | --- | --- | --- |
| 2   | 0   | 1   | 0   | 1   | 0   |
| 3   | 1   | 0   | 0   | 1   | 0   |
| 4   | 0   | 1   | 0   | 0   | 0    |

Elements $R^{-1}$: (4, 2), ( 6, 2), (3, 3), (6, 3), (4, 4)

|     | 2   | 3   | 4   |
| --- | --- | --- | --- |
| 3   | 0   | 1   | 0   |
| 4   | 1   | 0   | 1   |
| 5   | 0   | 0   | 0   |
| 6   | 1   | 1   | 0   |
| 7   | 0   | 0   | 0    |

## Question 11

Let $R_1$ and $R_2$ be the relations on a set $S = \{1, 2, 3, 4\}$ given by:

$R_1 = \{ (1, 1), (1, 2), (3, 4), (4, 2), (2, 4) \}$
$R_2 = \{(1, 1), (3, 2), (4, 4), (2, 2), (4, 2)\}$

1. Find the matrix representation $R_1$ and that of $R_2$.

$MR_1$

|     | 1   | 2   | 3   | 4   |
| --- | --- | --- | --- | --- |
| 1   | 1   | 1   | 0   | 0   |
| 2   | 0   | 0   | 0   | 1   |
| 3   | 0   | 0   | 0   | 1   |
| 4   | 0    | 1   | 0   | 0    |

$MR_2$

|     | 1   | 2   | 3   | 4   |
| --- | --- | --- | --- | --- |
| 1   | 1   | 0   | 0   | 0   |
| 2   | 0   | 1   | 0   | 0   |
| 3   | 0   | 1   | 0   | 0   |
| 4   | 0    | 1   | 0   | 1   |

2. Find the matrix of the intersection of both matrices in (1).

|     | 1   | 2   | 3   | 4   |
| --- | --- | --- | --- | --- |
| 1   | 1   | 0   | 0   | 0   |
| 2   | 0   | 0   | 0   | 0   |
| 3   | 0   | 0   | 0   | 0   |
| 4   | 0    | 1   | 0   | 0   |

3. Find the matrix of the union of both matrices in (1).

|     | 1   | 2   | 3   | 4   |
| --- | --- | --- | --- | --- |
| 1   | 1   | 1   | 0   | 0   |
| 2   | 0   | 1   | 0   | 1   |
| 3   | 0   | 1   | 0   | 1   |
| 4   | 0    | 1   | 0   | 1   |

4. List the elements of $R_1 \cap R_2$

$R_1 \cup R_2 = \{ (1, 1), (4, 2) \}$

5. List the elemnts of $R_1 \cup R_2$

$R_1 \cup R_2 = \{ (1, 1), (1, 2), (2, 2), (2, 4) (3, 2), (3, 4), (4, 2), (4, 4) \}$

## Question 12

Let $R$ be a relation on set $A$.

1. How can we quickly determine whether a relation R is reflexive by examining the matrix of $R$?
2. How can we quickly determine whether a relation R is symmetrix by examining the matrix of R?
3. How can we quickly determiner whether a relation R is anti-symmetric by examining the matrix of R?
