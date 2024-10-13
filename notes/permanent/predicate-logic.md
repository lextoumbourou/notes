---
title: Predicate Logic
date: 2023-10-13 00:00
summary: An extension of propositional logic that involves variables and quantifiers.
aliases:
  - First-Order Logic
tags:
  - DiscreteMath
  - MathematicalLogic
cover: /_media/predicate-logic-cover.png
---

An extension of [Propositional Logic](propositional-logic.md) that uses variables and quantifiers to represent and analyse [Statement](logical-statement.md).

## [Predicate](predicate.md)

Is a [Statement](logical-statement.md) that includes a variable.

* $P(x)$: "x is a prime number"

A predicate becomes a proposition when the variable are substituted for values.

* $P(2)$: "2 is a prime number" (True)

## [Quantifiers](../journal/permanent/logical-quantifiers.md)

Quantifiers describe how many of a thing there are.

### [Universal Quantifier](universal-quantifier.md)

Symbol: $\forall$

Means "For all" or "Every".

Example:

$\forall x, P(x)$: "For every x, x is a prime number"

### [Existential Quantifier](existential-quantifier.md)

Symbol: $\exists$

Means "There exists" or "Some".

Example:

$\exists x, P(x)$: "There exists an x such that x is a prime number"

## DeMorgan's Laws for negating quantifiers

 ∼[(∀x)P(x)] ≡ (∃ x)[∼P(x)]
 ∼[(∀x)P(x)] ≡ (∀x)[∼P(x)]
