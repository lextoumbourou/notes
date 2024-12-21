---
title: Separate Chaining
date: 2024-12-21 00:00
modified: 2024-12-21 00:00
status: draft
tags:
- DataStructures
- ComputerScience
---

Separate chaining is a collision resolution technique used in a [[Rehashing](rehashing.md)](hash-table.md) where each slot contains a linked list of elements that hash to that position. When a collision occurs (two keys hash to the same slot), the new element is simply added to the linked list at that position. This approach allows the hash table to handle an unlimited number of elements.