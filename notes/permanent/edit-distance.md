---
title: Edit Distance
date: 2024-12-22 00:00
modified: 2026-09-26 08:55
status: draft
---

**Edit Distance** is a way of measuring how different two strings are, by counting the minimum number of operations needed to transform one into the other.

The most common version is [Levenshtein Distance](Levenshtein%20Distance.md), which allows insertions, deletions and substitutions. [Damerau Levenshtein](damerau-levenshtein.md) distance also allows transpositions of adjacent characters.

For example, the Levenshtein distance between "kitten" and "sitting" is 3: substitute "k" for "s", substitute "e" for "i", then insert "g".
