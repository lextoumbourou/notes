---
title: Decrease and Conquer
date: 2024-02-21 00:00
modified: 2024-02-21 00:00
status: draft
---

**Decrease and Conquer** is a class of [Algorithm](algorithm.md) where a [Problem](problem.md) is solved by reducing to a smaller, similar problem, which we solve recursively.

Examples:

* [Binary Search](binary-search.md): to find a specific value in a sorted array, the algorithm repeatedly halved the part of the list that contains the item, decreasing the problem size at each step until an item is found or the list is empty.
* [Insertion Sort](insertion-sort.md): builds the final sorted array one item at a time. It takes an element from the list and places it in the correct place relative to the sorted part of the list.

Not to be confused with [Divide-and-Conquer](divide-and-conquer.md), which divides a problem into two or smaller, independent problems, solving each recursively and combining the solution.
