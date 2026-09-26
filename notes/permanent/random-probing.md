---
title: Random Probing
date: 2024-12-22 00:00
modified: 2026-09-26 08:55
status: draft
---

**Random Probing** is a [Hash Table](hash-table.md) collision resolution method. When a collision occurs, the element is placed in a randomly chosen slot instead of the next one.

In practice, the "random" sequence of slots is pseudo-random and determined by the key, so a lookup can follow the same sequence to find the element again.

Compare with [Linear Probing](linear-probing.md), which just checks the next slot, and [Separate Chaining](separate-chaining.md).
