---
title: Propositions
aliases: Statement
date: 2023-10-11 00:00
modified: 2025-01-01 00:00
tags:
  - DiscreteMath
  - MathematicalLogic
summary: Declarative sentences that are either true or false (but not both)
---

A **proposition** (also called a **statement**) is a declarative sentence with a truth value of either **true** or **false**, but not both.

Every proposition must be:

1. A complete sentence that makes a claim
2. Either true or false (but not both)
3. Definite in its truth value

For example, these are valid propositions:

* "I wrote this article on Thursday" (true)
* "I am 14 years old" (false)
* "1 + 1 = 3" (false)
* "Water boils at 100°C at standard atmospheric pressure" (true)
* "Every even integer is divisible by 2" (true)

These are not propositions:

* "What time is it?" - Questions aren't propositions as they don't make claims
* "Close the door" - Commands don't have truth values
* "What a beautiful day!" - Exclamations and opinions without clear criteria aren't propositions

Also, expressions with variables, where the value of the variable would affect the truth value, aren't propositions:

* $x + 2 = 5$  - Not a proposition, as its truth value depends on the value of $x$.

When a statement contains a variable whose truth value depends on that variable's value, it's called a [Predicate](predicate.md). 

For example:

* "I am $X$ years old"
* $Y + 1 = 5$
* "The speed of light in a vacuum is approximately $Z$ meters per second."

> [!QUESTION] Which of these is a valid **proposition**?
> a) I live in Australia.
> b) Z > 2
> c) Please don't do that anymore
> d) $1 + 2$
> e) $11 \times 11 = -1$
> > [!SUCCESS]- Answer
> > a) I live in Australia.
> > e) $11 \times 11 = -1$
> > Only (a) and (e) are valid propositions because:
> > * "I live in Australia" is a declarative statement that is either true or false
> > * $11 \times 11 = -1$ is a mathematical statement that is definitely false
> > * (b) contains a variable, making it a predicate
> > * (c) is a command
> > * (d) is an expression, not a statement

In propositional logic, we often assign propositions to variables, typically using lowercase letters $p$, $q$, $r$, $s$, or $t$:

* Let $p = \text{"It rained yesterday"}$
* Let $q = \text{"I am happy"}$

These variable assignments allow us to manipulate and analyse propositions using [Propositional Logic](propositional-logic.md), where we can combine simple propositions to form more complex ones.