---
title: Week 5 - Propositional Logic A
date: 2022-11-08 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
modified: 2023-04-08 00:00
---

## 3.101 Introduction to propositional logic

* [Propositional Logic](../../../../permanent/propositional-logic.md)
    * A branch of logic that is interested in studying mathematical statements.
    * The basis of reasoning and the rules used to construct mathematical theories.
    * Original purpose of propositional logic dates back to Aristotle. Used to model reasoning.
    * "An algebra of propositions".
        * Variables are unknown propositions not unknown real numbers.
        * Instead of `+`, `-`, `x`, `%`, the operators used are:
            * `and`
            * `or`
            * `not`
            * `implies`
            * `if`
            * `if and only if`
    * Used in:
        * computer circuit design.
        * programming languages and systems, such as language Prolog.
        * logic-based programming languages:
            * languages use "predicate logic", a more powerful form of logic that extends the capabilities of propositional logic

### 3.103 Propositions

* [Proposition](../../../../permanent/proposition.md)
    * A declarative sentence that is either true or false but not both.
    * The most basic element of logic.
    * Examples
        * London is the capital of the United Kingdom
            * A true proposition.
        * $1 + 1 = 2$
            * Another true proposition.
        * $2 < 3$
        * Madrid is the capital of France.
            * A false proposition.
        * 10 is an odd number
            * Another false proposition.
    * Examples that aren't propositions
        * $x + 1 = 2$
            * Since we don't know the value of x, we don't know if it's true or false.
        * $x + y = z$
        * What time is it?
            * Not a declarative sentence, so not a proposition.
        * Read this carefully.
        * This coffee is strong
            * Subjective meaning: not true or false.
* Propositional variables
    * Use variables for propositional shorthand.
    * Typically uses letter like: $p$, $q$, $r$
    * Examples
        * p: London is the capital of United Kingdom
        * q : 1 + 1 = 2
        * r : 2 < 3

### 3.105 Truth tables and truth sets

* [Truth Table](../../../../permanent/truth-table.md)
    * A tabular representation of possible combinations of constituent variables.
    * To construct the truth table for n propositions:
        * Create table with $2^n$ rows and n columns.
        * Fill the first n columns with all the possible combinations.
    * Example
        * Two propositional variables p and q:

            | p | q |
            | ----- | ----- |
            | FALSE | FALSE |
            | FALSE | TRUE |
            | TRUE | FALSE |
            | TRUE | TRUE |

* Truth Set
    * Let $p$ be a proposition of set $S$.
    * The truth set of $p$ is the set of elements of $S$ for which $p$ is true.
    * We use the capital letter to refer to truth set of a proposition.
        * Truth set of $p$ is $P$
    * Example
        * Let $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$
        * Let $p$ and $q$ be 2 propositions concerning an integer n in S, defined as follows:
            * $p$: $n$ is even
            * $q$: $n$ is odd
        * The truth set of $p$ written as $P$ is:
            * $P = {2, 4, 5, 8, 10}$
        * The truth of set q is:
            * Q = {1, 3, 5, 7, 9}

### 3.107 Compound propositions

* [Compound Statements](../../../../permanent/compound-statements.md)
    * Statements build by combining multiple propositions using certain rules.
* [Negation](../../../../permanent/logical-negation.md)
    * Not $p$: Defined by $\neg p$
    * "It is not the case that $p$"
    * The truth value of the negation of $p$, $\neg p$, is the opposite of truth value of $p$.
    * Example
        * $p$: John's program is written in Python
        * $\neg p$: John's program is not written in Python
* [Conjunction](../../../../permanent/conjunction.md)
    * Symbol: $\land$
    * $p$ and $q$
    * Let $p$ and $q$ be propositions.
    * Conjunction of $p$ and $q$ are denoted by $p \land q$
    * Conjunction is only true when both $p$ and $q$ are true. False if it isn't the case.
    * Example:
        * $p$: John's program is written in Python
        * $q$: John's program has less than 20 lines of code.
        * $p \land q$: John's program is written in Python and has < 20 lines of code.
* [Disjunction](../../../../permanent/logical-disjunction.md) * Symbol: $\lor$
    * $p$ or $q$
    * Let $p$ and $q$ be propositions.
    * The disjunction of $p$ and $q$ denoted by $p \lor q$ is only false when both $p$ and $q$ are false, otherwise true.
    * Example:
        * $p$: John's program is written in Python.
        * $q$: John's program is < 20 lines of code.
        * $p \lor q$: John's prgram is written in Python or has less than 20 lines of code.
* [Exclusive-Or](../../../../permanent/logical-xor.md)
    * Symbol: $\oplus$
    * $p$ or $q$ (but not both)
* Precedence of logical operations
    * To build complex compound propositions, we need to use parentheses.
    * Example:
        * $(p \lor q) \land (\neg r)$ is different from $p \lor (q \land \neg r)$
    * To reduce the number of parentheses, we can use order of precedence.

        | Operator | Precedence |
        | -------- | ---------- |
        | $\neg$ | 1 |
        | $\land$ | 2 |
        | $\lor$ | 3 |
