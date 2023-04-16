---
title: Set
aliases: Set Theory, Sets
date: 2023-01-15 00:00
tags:
  - SetTheory
  - DiscreteMath
summary: A set is a unique collection of well-defined objects.
modified: 2023-04-13 00:00
cover: /_media/set-theory.jpg
---

A set is a unique collection of well-defined objects.

Objects in a set are called **elements** or **members** of the set.

We often refer to sets in everyday language: you may have heard of a tea set, a drum set, a set of action figurines, etc. These terms refer to specific collections of objects. It is clear which objects in are members of each set and which are not.

Sets are commonly notated using curly braces. For example:

* The set of numbers from 1 to 3: $A = \{1, 2, 3\}$
* A set of words: $B = \{\text{hello}, \text{world}\}$

## Membership

The elements of a set must be well-defined, and there should be no ambiguity about whether something is a set member or not.

We use the $\in$ notation to describe if something is a set member:

* $1 \in A$
* $\text{hello} \in B$

We use the $\notin$ notation to describe if something is a not set member:

* $99 \notin A$
* $\text{goodbye} \notin B$

The symbols in Latex are:

* $\in$: `\in`
* $\notin$: `\notin`

A set can be a member of another set. Interestingly, this fact means that the definition of a "set" is circular, making it technically an undefined term.

## Uniqueness

Sets do not have duplicates; therefore:

$A = \{1, 1, 2\} = \{1, 2\}$

This property means sets are practically helpful for finding unique counts of things, and they often appear in programming for this purpose and many others.

## Subsets

If every element in set $A$ is in set $B$, we consider set $A$ to be a subset of $B$. The notation for subset is $A \subseteq B$. In LaTeX, we use: \subseteq

For example, the set of days on the weekend is a subset of the days in the week:

$\{\text{Sunday}, \text{Saturday}\} \subseteq \{\text{Sunday}, \text{Monday}, \text{Tuesday}, \text{Wednesday}, \text{Thursday}, \text{Friday}, \text{Saturday}\}$

## Supersets

Conversely, we consider $B$ to contain $A$, denoted as $B \supseteq A$. We say that $B$ is a superset of $A$.

If $A$ is not a subset of $B$, we write $A \nsubseteq B$. In LaTeX, it's: \nsubseteq

## Cardinality

The number of elements in a set is called [Cardinality](cardinality.md)

## Equal Sets

When 2 sets contain the same elements, we consider them **equal sets**: $A == B$.

## Empty Set

When a set has no elements it's called the **empty set**: $\emptyset$: $\emptyset = \{\}$. The empty set is a subset of every other set.

## Universal Set

A special set called the universal set $U$, is a set where every other set is a subset of $U$.

$A \subseteq U$ for every set $A$

## Disjoint Sets

A set may not have common elements. These are called disjoint sets. For example, a set of integers and a set of letters are disjoint sets.

## Power Set

The power set is a subset representing all subsets of set A, written as $P(A)$.

For example, the power set of $A = \{1, 2\}$:

$P(A) = \{\emptyset, \{1\}, \{2\}, \{1, 2\}\}$

## Infinite and finite sets

If a set has an infinite number of elements, we call it an infinite set.

Some [Special Infinite Set](special-infinite-sets.md) come up frequently in Math.

<hr>

Cover by <a href="https://unsplash.com/ko/@ahlianyq?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ahlianyq</a> on <a href="https://unsplash.com/photos/Cu80T4bZ0rI?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>.
