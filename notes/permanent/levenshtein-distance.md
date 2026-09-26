---
title: Levenshtein Distance
date: 2026-09-26 08:50
modified: 2026-09-26 08:50
status: draft
---

**Levenshtein Distance** is the minimum number of single-character edits (insertions, deletions or substitutions) needed to turn one string into another.

For example, the distance between "kitten" and "sitting" is 3: substitute k with s, substitute e with i, then insert g at the end.

It's the most common type of [Edit Distance](edit-distance.md), and is usually computed with dynamic programming in $O(mn)$ time for strings of length $m$ and $n$. [Damerau Levenshtein](damerau-levenshtein.md) distance extends it by also allowing transpositions of adjacent characters.
