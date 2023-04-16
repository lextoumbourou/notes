---
title: Week 12 - Recursion B
date: 2022-01-07 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
modified: 2023-04-08 00:00
---

## 6.201 Recursive definitions

* [Recursion](../../../../permanent/recursion.md)
    * When you must define a math object (set, func, sequence) in terms of object itself.
* Recursively defined functions
    * Recursively defined function f with domain $\mathbb{N}$ is defined by:
        * BASIS STEP: specify an initial value of function.
        * RECURSIVE STEP: rule for defining value of function when given integers, in terms of smaller integers.
    * We call this definition: recursive or inductive.
    * Defining a function $f(n)$ from set $\mathbb{N}$ to set $\mathbb{R}$ is same as sequence: $a_0, a_1$ ... where $\forall i \in \mathbb{N}$, $a_i \in \mathbb{R}$
    * Examples
        * Give recursive definition of sequence $\{a_n\}, n = 1, 2, 3, ...$ in following cases:
            * $a_n = 4n$
            * $a_n = 4^n$
            * $a_n = 4$
        * May be more than one correct answer to each sequence
            * 1. As each term in the sequence is greater than the previous term by 4, this sequence can be defined by setting $a_1 = 4$ and declaring that $\forall n \ge 1 \ a_{n+1} = 4 + a_n$
            * 2. As each term is 4 times its predecessor, this sequence can be defined as $a_1 = 4$ and $\forall n \ge 1 \ a_{n+1} = 4a_n$
                * $a_1 = 4$
                * $a_2 = 4^2 = 4 * 4 = 16$
                * $a_3 = 4^3 = 4 * 16 = 64$
                * $a_4 = 4^4 = 4 * 64 = 256$
            * 3. Sequence can be defined as $a_1 = 4$ and $\forall n \ge 1 \ a_{n+1} = a_n$
* Recursively defined sets
    * Like sequences, for some sets, defining recursively is easier than defining explicitly.
    * Sets can also be defined recursively, by defining two steps:
        * BASIS STEP: where we specify some initial elements.
        * RECURSIVE STEP: provide a rule for constructing new elements from those we already have.
    * Example
        * Consider the subset of S of the set of integers recursively defined by:
            * 1. BASIS STEP: $4 \in S$ (4 is an element of S)
            * 2. RECURSIVE STEP: if $x \in S$ and $y \in S$ then $x + y \in S$
        * We will be shown later how we can prove that set S is the set of all positive integers that are multiples of 4.
* Recursive algorithms
    * Algorithm
        * A finite sequence of precise instructions for performing a computation or solving a problem.
        * An algorithm is considered recursive if it solves a problem by reducing to an instance of the same problem with smaller input.
    * Example
        * Let's give a recursive algorithm for computing $n!$ where n is a nonnegative integer:
            * n! can be recursively defined by the following 2 steps:
                * BASIS STEP: $0! = 1$
                * RECURSIVE STEP: $n! = n (n - 1)!$ where $n$ is a positive integer.
            * The pseudocode of the algo can be formalised as:

        ```
        procedure factorial(n: nonnegative integer) {
            if n = 0 then return 1
            else
            return n factorial (n - 1)
        }
        ```

## Lesson 6.204 Recurrence relations

* Recurrence Relation
    * An equation that defines a sequence based on a rule which gives the next term as a function of the previous term.
* Infinite Sequence
    * Function from the set of positive integers to set of real numbers.
* It can be useful to formalise the problem as sequence before solving it.
* Example: [[Towers of Hanoi]]
    * ![[week-12-hanoi-tower.png]]
      * Want to get discs from spoke A to C.
      * Can only move one disk at a time.
      * You cannot place a larger disc on a smaller one.
      * It is an example of a recursive function.
      * Number of moves:
          * Let $a_n$ be min number of moves to transfer n from one spoke to another.
          * To move n discs from A to C, we must move n-1 discs from A to B by $a_{n-1}$ moves.
          * Then, move last (and largest) disc from A to C by 1 move.
          * Then, remove the n-1 discs again from B to C by $a_{n-1}$ moves.
          * Thus, total moves is $a_n = 2a_{n-1} + 1$
* [Linear Recurrence](permanent/linear-recurrence.md)
    * In which each term of a sequence is a linear function of earlier terms in the sequence.
    * Two types of linear recurrence:
        * Linear homogeneous recurrence:
            * formalised as $\mathbf{a_n} = c_1\mathbf{a_{n-1}} + c_2\mathbf{a_{n-2}}+... + c_k\mathbf{a_{n-k}}$
            * where $c_1, c_2, ..., c_k \in \mathbb{R}$, and k is the degree of relation.
        * Linear non-homogeneous recurrences:
            * formalised as $\mathbf{a_n} = c_1\mathbf{a_{n-1}} + c2\mathbf{a_{a-2}} + ... ck\mathbf{a_{n-k}} + f(n)$
            * where $c_1, c_2, ..., c_k \in \mathbb{R}$, and f(n) is a function.
            * Depending only on n, and k is the degree of relation.
    * Example: first order recurrence
        * Consider:
            * a country has 50 million people that:
                * has population growth rate (birth rate - death rate) of 1% per year.
            * receives 50,000 immigrants per year.
        * Question: find this country's population in 10 years.
        * This case can be modelled as the following first-order linear recurrence:
            * where $a_n$ is the population in n years from now.
            * $\forall n \in \mathbb{N}$, $a_{n+1}$ is expressed as $a_{n+1} = 1.01 \mathbf{a_n} + 50,000$
            * $a_0 = 50,000,000$.
    * Example: 2nd order recurrence
        * Consider:
            * 0, 1, 1 2, 3, 5, 8, 13, 21, 34, ...
            * where each number is found by adding up the two numbers before it.
        * We can model the sequence as 2nd-order linear recurrence:
            * $a_n = a_{n-1} + a_{n-2}$
            * $a_0 = 0$
            * $a_1 = 1$
        * This is called Fibonacci sequence.
* Arithmetic sequences
    * We consider a sequence arithmetic if distance between consecutive terms is a constant c
    * $\forall n$, $a_{n+1}$ is expressed as $a_{n+1} = a_n + c$ and $a_n = a$
    * Example:
        * Sequence 2, 5, 8, 11, 14, ... is arithmetic with initial term of $a_0 = 2$ and a common difference of 3.
        * 30, 25, 20, 15, ... is arithmetic with initial term of $a_0 = 30$ and common difference of -5.
* Geometric sequences
    * Sequence is geometric if ratio between consecutive terms is a constant r.
    * $\forall n$, $\mathbf{a_{n+1}}$ is expressed as $a_{n+1} = r \ \mathbf{a_n}$ and $\mathbf{a_0} = a$
        * Example:
            * Sequence 3, 6, 12, 24, 48, ... is geometric with an initial term of $a_0 = 3$ and a common ratio of 2.
            * 125, 25, 5, 1, 1/5, 1/25, ... is geometric with initial term of $a_0 = 125$ and common ratio of 1/5.
* Divide and conquer recurrence
    * Divide and conquer algorithm consists of three steps:
        * dividing problem into smaller subproblems.
        * solving (recursively) each subproblem.
        * combining solutions to find a solution to original problem.
        * Example:
            * Consider the problem of finding minimum of a sequence $\{a_n\}$ where $n \in \mathbb{N}$
                * if n=1, the number is itself min or max.
                * if n > 1, divide the numbers into 2 lists.
                * order the sequence.
                * find min and max in first list.
                * find min and max in second.
                * infer the min and max of entire list.

### Lesson 6.206 Solving recurrence relations

* Solving linear recurrence
    * Let $a_n = c_1 \ \mathbf{a_{n-1}} + c_2 \ \mathbf{a_{n-2}}  + ... + c_k \ \mathbf{a_{n-k}}$ be a linear homogeneous recurrence.
    * If a combination of the geometric sequence $a_n = r^{n}$ is a solution to this recurrence, it satifies $r^n = c_1 \ r^{n-1} + c_2 \ r^{n-2} ... c_k \ r^{n-k}$
    * By dividing both sides by $r^{n-k}$, we get: $\mathbb{r^{k}} = c_1 \ \mathbb{r^{k-1}} + c_2 \ \mathbb{r^{k-2}} ... c_k$
    * We call this equation the Characteristic Equation.
        * Solving this equation is first step towards finding a solution to linear homogeneous recurrence:
            * If $r$ is a solution equation with multiplicity p, then the combination $(\lambda +\beta n + yn^2 + ... + \mu n^{p-1})\mathbf{r^{n}}$ satisifes the recurrence.
* Solving Fibonacci recurrence
    * Let's consider solving the Fibonacci recurrence relation: $f_n = f_{n-1} + f_{n-2}$, with $f_0 = 0$ and $f_1 = 1$
        * Solution:
            * The characteristic equation of the Fibonacci recurrence relation is: $r^2 - r - 1 = 0$
            * It has 2 distinct roots, of multiplicity 1:
                * $r_1 = (1 + \sqrt{5}) / 2$ and $r_2 = (1 - \sqrt{5}) / 2$
            * So, $f_n = \alpha_1 \ r_1^{n} + \alpha_2 \ r_2^n$ is a solution.
        * To find $\alpha_1$ and $\alpha_2$ we need to use the initial conditions.
            * From:
                * $f_0 = \alpha_1 + \alpha_2 = 0$
                * $f_1 = \alpha_1(1 + \sqrt{5})/2 + \alpha_2 (1 - \sqrt{5})/2 = 1$
            * By solving the 2 equations, we can find $\alpha_1 = 1/\sqrt{5}$ and $\alpha_2 = -1 / \sqrt{5}$
        * The solution is then formalise as:
            * $f_n = 1 / \sqrt{5} . (((1 + \sqrt{5}) / 2)^{n})  - 1/\sqrt{5} . ((( 1 - \sqrt{5})/2)^{n})$
* Example 2
    * Consider another sequence
        * $a_n = -3a_{n-1} - 3a_{n-2} - a_{n-3}$
        * $a_0 = 1, a_1 = -2 \text{ and } a_2 = -1$
        * Solution:
            * The characteristic equation of this relation is: $r^3 + 3r^2 + 3r + 1 = 0$
            * It has a distinct root, whose multiplicity is 3, $r_1 = -1$
            * So, $a_n = (\alpha_0 + \alpha_1n + \alpha_2n^2)r_1^2$ is a solution
        * To find $\alpha_0, \alpha_1$ and $\alpha_2$ we need to use initial conditions.
            * From:
                * $a_0 = a_0 = 1$
                * $a_1 = -(\alpha_0 + \alpha_1 + \alpha_2) = -2$
                * $a_2 = (\alpha_0 + 2\alpha_1 + 4\alpha_2) = -1$
            * We can find $\alpha_1 = 3, \alpha_2 = -2$
        * The solution is then formalised as: $a_n = (1 + 3n - 2n^2)(-1)^{n}$
* Using strong induction to solve recurrence
    * It is sometimes easier to solve recurrence relation solutions using strong induction.
    * Example:
        * Prove:
            * P(n): the sequence f_n = 1/\sqrt{5}(r_1^{n} -r_2^{n}) verifies the Fibonacci recurrence, where:
                * $r_1 = (1 + \sqrt{5}) / 2$
                * $r_2 = (1 - \sqrt{5}) / 2$ are the roots of $r^2 - r - 1 = 0$$
            * First, verify for P(2):
                * $f_1 + f_0 = 1 / \sqrt{5}(r_1 - r_2) = 1/\sqrt{5} (\sqrt{5}) = 1 = f_2$
                * because $f_2 = 1/\sqrt{5}(r_1^2 - r_2^2) = 1$
                * which verifies the initial condition.
            * Let $k \in \mathbb{N}$, where P(2), P(3) ... P(k) are all true.
            * Let's verify $P(k+1)$:
                * $f_n + f_{n-1} = \frac{r_1^n - r_2^2}{\sqrt{5}} + \frac{r_1^{n-1}-r_2^{n-2}}{\sqrt{5}} = r_1^{n-1}(r_1 + 1) / \sqrt{5} + r_2^{n-2}(r_2 + 1) / \sqrt{5} = r_1^{n-1} * r_1^{2} + r_2^{n-1}* r_2^{2} = r_1^{n+1} + r_2^{n+1}$ which equals $f_{n + 1}$
            * We can conclude that $P(k+1)$ is true and the strong induction is verified.

## Assignment 6.209 - Induction and Recursion

Use mathematical induction to prove that $\forall n \in N$

$P(n) : 1 · 2 · 3 + 2 · 3 · 4 + ···  + n(n + 1)(n + 2) = n(n + 1)(n + 2)(n + 3)/4$

### Answer

1. BASIS STEP

The basis step $P(1)$ is true because:

$1 * 2 * 3 = \frac{1(1 + 1)(1 + 2)(1 + 3)}{4}$

2. INDUCTION STEP

Let $k$ be an arbitrary element, where $P(k)$ is true.

$P(k) = 1 \cdot 2 \cdot 3 + 2 \cdot 3 \cdot 4 + ... + k(k + 1)(k + 2) = \frac{k(k+1)(k+2)(k+3)}{4}$

Verify $P(k + 1)$

$P(k + 1) = 1 \cdot 2 \cdot 3 + 2 \cdot 3 \cdot 4 + ... k(k + 1)(k + 2) + (k + 1)(k + 2)(k + 3)$
$P(k + 1) = \frac{k(k+1)(k+2)(k+3)}{4} + (k+1)(k + 2)(k + 3) = \frac{(k + 1)(k + 2)(k + 3)(k + 4)}{4}$

Which means that $P(k + 1)$ is true and the induction step is verified.

## Problem Sheet

## Question 1

Prove that the sum of any even integers is even. In an other way show that:

$\forall n, m \in \mathbb{Z}$, if $n$ and $m$ are even numbers then $n+m$ is also an even number.

**Proof**

Let $n, m \in \mathbb{Z}$ and assume that $n$ and $m$ are even.
We need to show that n + m is also even.
n and m are two even integers, it follows by definition of even numbers that there exists 2 integers $i$ and $j$ such that $n = 2i$ and $m = 2j$.
Thus, $n + m = 2i + 2j = 2(i + j)$.
Hence, there exists an integer $k = i + j$ such that $n + m = 2k$, it follows by definition of even numbers that n + m is even.

## Question 2

Use direct proof to show that: $\forall n, m \in \mathbf{Z}$, if $n$ is an even number and m is an odd number, then 3n + 2m is also an even number.

**Proof**

Let $n, m \in \mathbb{Z}$ and assume that $n$ is an even number and $m$ is an odd number. We need to show that 3n + 2m is also an even number.

Assume that $n$ is even and $m$ is odd, this implies that there exists 2 integers $i, j \in \mathbb{Z}$ such that: $n = 2i$ and $m = 2j + 1$.

Thus, 3n + 2m = 3(2i) + 2(2j + 1) = 2(3i) + 2(2j + 1) = 2(3i + 2j + 1)

Hence, there exists an integer k = (3i + 2j + 1) such that 3n + 2m = 2k.

By the definition of even numbers, we can see 3n + 2m number is even.

## Question 3

Prove that the sum of any 2 odd integers is odd. In an other way show that: $\forall n, m \in \mathbb{Z}$ if $n$ and $m$ are odd numbers then $n + m$ is an even number.

**Proof**

Let $n, m \in \mathbb{Z}$ and assume that $n$ is odd and $m$ is odd. We need to show that $n + m$ is even.

$n$ and $m$ are odd, which follows by definition of odd numbers that there exists 2 integers $i$ and $j$ such that $n = 2i + 1$ and $m = 2j + 1$

Thus, $n + m = 2i + 1 + 2j + 1 = 2i + 2j + 2 = 2(n + m + 1)$

Hence, there is an integer $k = (n + m + 1)$ such that $n + m = 2k$, which proves $n + m$ is even by the definition of even numbers.

### Question 4

Show that for any odd number integer $n$, $n^2$ is also odd.

In another way show that: $\forall n \in \mathbb{Z}$ if $n$ is odd then $n^2$ is odd.

**Proof**

Let $n \in \mathbb{Z}$ and assume that $n$ is an odd number. We need to show that $n^2$ is also odd, which means show there's an integer $k$ such that $n^2 = 2k + 1$.

By definition of odd, $n$ is odd which means there exists an integer $i$ such that $n = 2k + 1$.

It follows that $n^2 = (2i+1)^2 = (2i + 1)(2i + 1) = 4i^2 + 4i + 1 = 2(2i^2 + 2i) + 1$

$2i^2 + 2i$ is an integer as it's the sum of products of integers.

Therefore, there exists $k = 2i^2 + 2i$ such that $n^2 = 2k + 1$.

It follows by definition of odd numbers that $n^2$ is an odd number.

---
