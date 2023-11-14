---
title: Week 11 - Mathematical Induction A
date: 2022-12-19 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
modified: 2023-04-08 00:00
---

## 6.101 Introduction to proofs

* [Proof](../../../../permanent/mathematical-proof.md)
    * A proof is a valid argument, that is used to prove the truth of a statement.
    * To build a proof, we need to use things previously introduced:
        * variables and predicates.
        * quantifiers.
        * laws of logic.
        * rules of inference.
    * Some terminology used:
        * Theorem
            * formal statement that can be shown to be true.
        * Axiom
            * a statement we assume to be true to serve as a premise for further arguments.
        * Lemma
            * a proven statement used as a step to a larger result rather than as a statement of interest by itself.
* Formalising a theorem
    * Consider statement S: "There exists a real number between any two not equal real numbers"
    * S can be formalised as: $\forall x, y \in \mathbb{R}$ if $x < y$ then $\exists z \in \mathbb{R}$ where $x < z < y$.
    * S is an example of a theorem.
* Direct proof
    * A direct proof is based on showing that a conditional statement: $p \rightarrow q$ is true.
    * We start by assuming that p is true and then use: Axioms, **definitions** and Theorems, together with **rules of inference** to show that q must also be true.
    * Example
        * "There exists a real number between any two not equal real numbers"
        * Proof:
            * Let $x, y$ be arbitrary elements in $\mathbb{R}$
            * Let's suppose $x < y$
            * Let $z = (x + y) / 2$
            * $z \in \mathbb{R}$, satisfying $x < z < y$
        * Therefore, using the universal generalised rule, we can conclude that: $\forall x, y \in \mathbb{R}$ if $x < y$ then $\exists z \in \mathbb{R}$ where $x < z < y$.
* Proof by Contrapositive
    * A **proof by contrapositive** is based on the fact that proving the conditional statement $p \rightarrow q$ is equivalent to providing its contrapositive $\neg q \rightarrow \neg p$
        * We start by assuming that $\neg q$ is true and then use axioms, definitions and theorems, together with **rules of inference**, to show that $\neg p$ must be true.
    * Example
        * Proof of the theorem:
            * If $n^2$ is even then n is even
            * Direct proof:
                * Let $n \in Z$. If $n^2$ is even then $\exists k \in Z$, $n^2 = 2k$
                * Then $\exists k \in Z$, $n = \pm \sqrt{2k}$
                    * From this equation it doesn't seem intuitive to prove that n is even.
            * Proof by contraposition:
                * Let's supposed n is odd.
                * Then $\exists k \in Z$, $n = 2k + 1$
                * Then $\exists k \in Z$, $n^2 = (2k + 1)^2 = 2(2k^2 + 2k) + 1$
                * Then $n^2$ is also odd.
                * We have succeeded in providing the contrapositive: if n is odd then $n^2$ is odd.
* Proof by contradiction
    * A proof by contradiction is based on assuming that the statement we want to prove is false, and then showing that this assumption leads to a false proposition.
    * We start by assuming that $\neg p$ is true and then use: [Axiom](Axiom)s, definitions and [Theorem](Theorem)s, together with **rules of inference** to show that $\neg p$ is also false. We can conclude that it was wrong to assume that p is false, so it must be true.
    * Example
        * Let's give a direct proof of the theorem: "There are infinitely many prime numbers"
        * Proof
            * Let's suppose that there are only finitely many prime numbers
            * Let's list them as $p_1, p_2, p_3, ..., p_n$ where $p_1 = 2, p_2 = 3, p_3 = 5$ and so on.
            * Let's consider the number c = $p_1 \ p_2 \ p_3 ... p_n + 1$, the product of all prime numbers, plus 1.
            * Then, as c is a natural number, it has at least one prime divisor.
            * Then $\exists k \in \{1 ... n\}$ where $p_k | c$ (where $p_k$ divides $c$).
            * Then $\exists k \in \{1 ... n\}, \exists d \in N$ where $dp_k = c = p_1 \ p_2 \ p_3 ... p_n  + 1$
            * Then $\exists k \in \{1 ... n\}$, $\exists d \in N$ where $d = p_1 p_2 ... p_{k + 1} ... p_n + \frac{1}{p_k}$
            * Then, $\frac{1}{p_k}$, in the expression above, is an integer, which is a contradiction.

## 6.103 The principle of mathematical induction

* [Mathematical Induction](../../../../permanent/induction.md)
    * Can be used to prove that a propositional function $P(n)$ is true for all positive integers.
    * The rule of inference:
        $P(1) \text{ is true}$
        $\forall k (P(k) \rightarrow P(k + 1))$
        $\therefore \forall n P(n)$
        * Intuition:
            * P is true for 1.
            * Since P is true for 1, it's true for 2.
            * Since P is true for 2, it's true for 3.
            * And so on...
            * Since P is true for n-1, it's true for n ...
        * In other words:
            * The base case shows that the property initially holds true.
            * The induction step shows how each iteration influences the next one.
    * Structure of induction
        * In order to prove that a propositional function P(n) is true for all, we need to verify 2 steps:
            * 1. **BASIS STEP**: where we show that P(1) is true.
            * 2. **INDUCTION STEP**: where we show that for $\forall k \in \mathbb{N}$: if $P(k)$ is true called **inductive hypothesis**, then $P(k + 1)$ is true.
    * Some uses of induction
        * Mathematical induction can be used to prove P(n) is true for all integers greater than a particular integer, where P(n) is a propositional function. Might cover multiple cases like:
            * Proving formulas
            * Proving inequalities
            * Proving divisibility
            * Providing properties of subsets and their cardinality.

## 6.106 Proof by induction

* [Proof By Induction](../../../../permanent/proof-by-induction.md)
    * Proving formulas
        * Proving a simple formula formalised as the propositional function, $P(n): 1 + 2 + 3 + ... + n = n (n + 1) / 2$
        * The sum of 1 to n is equal to n multipled by n plus 1 divided by 2, for all n in N.
    * In order to prove that a propositional function $P(n)$ is true for all N, we need to verify 2 step:
        * 1. BASIS STEP: where we show that P(1) is true.
        * 2. INDUCTIVE STEP: where we show that for $\forall k \in \mathbb{N}$: if $P(k)$ is true, called **inductive hypothesis**, then P(k + 1) is true.
    * 1. BASIS STEP: The basis step, P(1) reduces to 1 = $P(1) = 1 (2) / 2 = 1$
    * 2. INDUCTIVE STEP:
        * Let $\forall k \in \mathbb{N}$
        * If the inductive hypothesis P(k) is true:
            * we have $1+2+3+...+k = k(k+1)/2$
            * then, $1+2+3+...+k+(k+1)$
            * $= k(k+1) / 2+(k+1)$
            * $= (k (k + 1) + 2(k + 1)) / 2$
            * $= (k + 1) ((k + 1) + 1) / 2$
            * which verifies, $P(k+1)$
* Proving inequalities
    * We may also use math induction to prove an inequality that holds for all positive integers greater than a particular positive integer.
    * Consider proving the propositional function $P(n): 3^n < n!$ if n is an integer greater than or equal to 7.
        * 1. BASIS STEP: The basis step, $P(7)$ reduces to $3^7 < 7!$ because 2187 < 5040.
        * 2. INDUCTIVE STEP:
            * Let $k \in \mathbb{N}$ and $k \ge 7$
            * If inductive hypothesis $P(k)$ is true:
                * then $3^{k+1} = 3 * 3^{k} < (k + 1) * k! = (k + 1)!$ which verifies $P(k + 1)$ is true.
* Proving divisibility
    * We can use math induction to prove divisibility that holds for all positive integers greater than positive integer.
    * Consider proving the prop function $P(n): \forall n \in \mathbb{N} \ 6^{n} + 4$ is divisible by 5.
    * Example
        * 1. BASIS STEP: The basis step, P(0), reduces to $6^{0} + 4$ is divisible by 5, because $6^{0} + 4 = 5$
        * 2. INDUCTION STEP:
            * Let $k \in \mathbb{N}$
            * If inductive hypothesis P(k) is true:
                * then, $6^{k} + 4 = 5p$ where $p \in \mathbb{N}$
                * then, $6{k+1} + 4 = 6 * (5p - 4) + 4 = 30p - 20$
                * $= 5(6p - 4)$ which is divisible by 5 and verifies $P(k + 1)$ is true.
* Incorrect Induction
    * Consider the statement of the following incorrect induction: $P(n): \forall n \in \mathbb{N} \sum^{n-1}_{i=0} 2^{i} = 2^{n}$
    * Proof:
        * Let $k \in \mathbb{N}$. Let's suppose the inductive hypothesis $P(k)$ is true, which means: $\sum^{k-1}_{i=0} 2^{i} = 2^{k}$
        * Let's examine $P(k + 1)$
        * $\sum^{k}_{i=0} 2^{i} = \sum^{k-1}_{i=0} 2^{i} + 2^{k} = 2^{k} + 2^{k} = 2^{k + 1}$
        * This means that $P(k + 1)$ is also true and verifies the induction step.
    * Even though we have been able to prove induction step, let's prove statement: $\forall n \in \mathbb{N}$ $\sum^{n-1}_{i=0} 2^{i} = 2^{n}$ is FALSE.
        * For example, $2^{0} + 2^{1} = 3$ which is different from $2^2$
    * Our reasoning seemed correct but we didn't verify the base case and have made **false assumptions**.
    * In other words, as we saw in propositional logic, false assumptions imply false conclusions.
    * To avoid this situation, we need to make sure both the base case and induction step are verified.

## 6.108 Strong induction

* [Strong Induction](../../../../permanent/strong-induction.md)
    * Can be formalised with rule of inference:
        * $P(1)$ is true.
        * $\forall k \in \mathbb{N} \ P(1), P(2) ... P(k) \rightarrow P(k+1)$
        * $\therefore \forall n \in \mathbb{N}, P(n)$
    * Sometimes easier to prove statements using strong induction than other methods.
    * Strong induction is sometimes called: "the 2nd principal of mathematical induction" or "complete induction".
    * Example:
        * Prove propositional function, P(n): $\forall n \in \mathbb{N}$ and $n \ge 2, n$ is divisible by prime number.
        * To prove need to verify 2 steps:
            * 1. **BASIS STEP**: $P(2)$ reduces to 2, which is divisible by prime number because 2 is a prime number and divides itself.
            * 2. **INDUCTIVE STEP**:
                * Let $k \in \mathbb{N}$ greater than 2.
                * If inductive hypothesis, $P(k)$ is true:
                    * Let's also assume $P(2) ... P(k+1)$ is true. Then, $\forall m \in \mathbb{N}$ and $2 \le m \le k+1 : \exists p$
                    * Then, $\forall m \in \mathbb{N}$ and $2 \le m \le k + 1$: $\exists p$ is a prime number dividing m.
                    * We have two cases:
                        * k + 2 is a prime number, in which case it is trivially divisible by itself.
                        * k + 2 is not a prime number, in which case $\exists m$ dividing $k + 2$
                        * as $2 \le m \le k + 1$, $\exists p$ is a prime number dividing m. p also divides k+2
                        * Which verifies P(k + 2) is true and proves the strong induction.
* [Well-Ordering Property](../../../../permanent/well-ordering-property.md)
    * The well-ordering property is an axiom about $\mathbb{N}$ that we assume to be true. The axioms about $\mathbb{N}$ are the following:
        * 1. The number 1 is a positive integer.
        * 2. If $n \in \mathbb{N}$ then $n + 1$, the successor of n, is also a positive integer.
        * 3. Every positive integer other than 1 is the successor of a positive integer.
        * 4. The well-ordering property: every nonempty subset of positive integers has at least one element.
    * The well-ordering property can be used as a tool in building proofs.
    * Example
        * Let's reconsider the earlier statement P(n): $\forall n \in \mathbb{N}$ and $n \ge 2$, n is divisible by a prime number.
        * Proof
            * Let $S$ be the set of positive integers greater than 1 with no prime divisor.
            * Suppose $S$ is nonempty. Let n be its smallest element.
            * n cannot be a prime, since n divides itself and if n were prime, it would be its own prime divisor.
            * So n is composite: it must have a divisor d with 1 < d < n. Then, d must have a prime divisor d with 1 < d < n. Then, d must have a prime divisor (by the minimality of n), let's call it p.
            * Then p / d and d / n, so p/n, which is a contradiction.
            * Therefore S is empty, which verifies P(n).
* Equivalence of the three concepts
    * We can prove the following statements:
        * mathematical induction -> the well-ordering property.
        * the well-ordering property -> strong induction.
        * strong induction -> mathematical induction.
    * That is, the principles of mathematical induction, strong induction and well-ordering are all equivalent.
        * The validaity of each of the 3 proof techniques implies the validity of the other 2 technicques.
