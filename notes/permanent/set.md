---
title: Set
aliases: Set Theory, Sets
date: 2022-10-16 00:00
tags:
  - SetTheory
  - DiscreteMath
summary: Set is a unique collection of well-defined things
status: draft
---

A set is a unique collection of things.

Things in a set are called **elements** or **members** of the set.

We talk about sets in everyday English: you may have heard of a kitchen set, a drum set, a set of action figurines, etc. These refer to a particular collection of objects. It is clear which objects in our house are members of that set and which are not.

Sets are commonly notated using brackets as follows:

A set of numbers ranged 1-3: $\{1, 2, 3\}$
A set of words: {hello, world}

## Set Membership

The elements of a set must be well-defined, and there should be no ambiguity about whether something is a set member or not.

We use the $\in$ notation to describe if something is a set member.

$a \in A$
$1 \in B$

We use the $\notin$ notation to describe if something is a set member.

The symbols in Latex are:

- $\in$: `\in`
- $\notin$: `\notin`

A set can be a member of another set. This fact means that the definition of a "set" is circular, making it technically an undefined term.

## Uniqueness

Sets do not have duplicates, therefore:

$A = \{1, 1, 2\} = \{1, 2\}$

This property means they are practically helpful for finding unique counts of things, and they come up frequently in programming for this purpose and many others.

## Subsets

If every element in set $A$ is in set $B$, we consider set $A$ to be a subset of $B$. The notation for **subset** is $A \subseteq B$. In Latex, we use: `\subseteq`

For example, the set of days on the weekend is a subset of the days in the week:

$\{\text{Sunday}, \text{Saturday}\} \subseteq \{\text{Sunday}, \text{Monday}, \text{Tuesday}, \text{Wednesday}, \text{Thursday}, \text{Friday}, \text{Saturday}\}$

## Supersets

In the inverse, we consider $B$ to contain $A$, denoted as $B \supseteq A$. Or we say that $B$ is a superset of $A$.

If A is **not a subset** of B, we write $A \nsubseteq B$. In Latex, it's: `\subsetneq`

## Cardinality

[Cardinality](cardinality.md)

## Equal Sets

When 2 sets contain the same elements, we consider them **Equal Sets**: $A == B$.

## Empty Set

When a set has no elements: **Empty Set**: $\emptyset$: $\emptyset = \{\}$. The empty set is a subset of every other set.

## Universal Set

A special set called the universal set $U$, is a set where every other set is a subset of $U$. 

$A \subseteq U$ for every set $A$

## Disjoint Sets

A set may not have common elements. These are called disjointed sets. For example, a set of integers and a set of letters are disjointed sets.

## Power Set

The power set is a subset representing all subsets of set A, written as $P(A)$.

For example, the power set of $A = \{1, 2\}$:

$P(A) = \{\emptyset, \{1\}, \{2\}, \{1, 2\}\}$

## Infinite and finite sets

If a set has an infinite number of elements, we call it an infinite set.

Some special infinite sets come up all the time in math:

* $\mathbf{Z}$ = set of integers = $\{...,2,−1,0,1,2, ...\}$
* $\mathbb{N}$ = $\mathbf{Z^{+}}$ = set of positive integers = $\{1,2,3,...\}$
* $\mathbf{Z−}$ = set of negative integers = $\{. . . , −3, −2, −1\}$
* $\mathbf{W}$ = set of whole numbers = $\{0,1,2,3,...\}$
* $\mathbf{Q}$ = set of rational numbers = $\{p/q|p, q ∈ \mathbf{Z} ∧ q \neq 0\}$
* $\mathbb{R}$ = set of real numbers
* $\mathbb{R^{+}}$ = set of positive real numbers = $\{x ∈ R|x > 0\}$
* $\mathbb{R}−$ = set of negative real numbers = $\{x ∈ R|x < 0\}$