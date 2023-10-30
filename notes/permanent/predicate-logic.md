---
title: Predicate Logic
date: 2023-04-09 00:00
summary: An extension of propositional logic that involves variables and quantifiers.
aliases:
  - First-Order Logic
status: draft
tags:
  - DiscreteMath
---

An extension of [Propositional Logic](propositional-logic.md) that uses variables and quantifiers to represent and analyse statements.

## [Predicate Logic](predicate-logic.md)

A statement involving variables.

A predicate becomes a proposition when specific values are substituted for its variables.

Example:

- $P(x)$: "x is a prime number"
- $P(2)$: "2 is a prime number" (True)

## [Quantifiers](../journal/permanent/logical-quantifiers.md)

Quantifiers describe how many of a thing there are.

- Universal Quantifier: ∀
- Symbol: $\forall$

Means "For all" or "Every".

Example:

$\forall x, P(x)$: "For every x, x is a prime number" (False, since not every number is prime)

- Existential Quantifier: ∃
- Symbol: $\exists$

Means "There exists" or "Some".

Example:

$\exists x, P(x)$: "There exists an x such that x is a prime number" (True, since there are prime numbers)

## Domain

The set of all possible values that a variable can take. It's essential to specify the domain because the truth value of a statement can change depending on the domain.

Example:

Domain of x is all natural numbers: N.
Bound and Free Variables
Bound Variable: A variable that is under the scope of a quantifier.
In $\forall x, P(x)$, x is bound.
Free Variable: A variable not bound by a quantifier.
In $P(x)$, x is free.
Negating Quantifiers
$\neg \forall x, P(x) \equiv \exists x, \neg P(x)$
(Not all x are P is equivalent to There exists an x that is not P)

$\neg \exists x, P(x) \equiv \forall x, \neg P(x)$
(There doesn't exist an x that is P is equivalent to All x are not P)

Multiple Quantifiers
Statements can have multiple quantifiers.

Example:

$\forall x \exists y, Q(x, y)$: "For every x, there exists a y such that Q(x, y) is true."
Rules of Inference
Predicate logic also has rules of inference, just as propositional logic does, but these also involve quantifiers.

For example, Universal Instantiation (UI) allows us to move from a universal statement to a particular instance:

From $\forall x, P(x)$, infer $P(c)$ for any element c in the domain.
Predicate Logic vs Propositional Logic
Predicate logic is more powerful than propositional logic. While propositional logic can handle statements that are true or false, predicate logic can handle statements with variables, making it capable of expressing more complex statements and relations.

Note: Predicate Logic (also called First-Order Logic) can get complex, especially when multiple quantifiers or nested quantifiers are involved. This is a basic introduction, and there's much more to explore, including predicate calculus, relations, and functions within predicate logic.
