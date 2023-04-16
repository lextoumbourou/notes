---
title: Powerset
date: 2023-04-09 00:00
status: draft
---

A Powerset is [Set](set.md) containing all possible subsets of another set. Represented by $P(S)$.

Examples:

$S = \{1, 2, 3\}$
$P(S) = \{ \emptyset, \{1\}, \{2\},\{3\},\{1, 2\},\{1, 3\}, \{2, 3\}, \{1, 2, 3\} \}$

$S = \{ a, b \}$
$P(S) = \{\emptyset, \{a\}, \{b\}, \{a, b\} \}$

$\{a\} \in P(S)$

Powerset of the empty set is set containing empty set: $P(\emptyset) = \{ \emptyset \}$

[Cardinality](cardinality.md) of a powerset of S is $2^|S|$

$|P(S)| = 2^{|S|}$
*$S = \{1, 2, 3\}$, $|S| = 3$, $|P(S)| = 2^3 = 6$
$P(S) = \{ \emptyset, \{1\}, \{2\}, \{3\}, \{1, 2\}, \{2, 3\} \}$
