---
title: "Week 5: Hash Maps and Edit Distance"
date: 2016-10-04 00:00
category: reference/moocs
status: draft
parent: data-structures-optimizing-performance
---

## Core: Hash Tables

* Hash table key idea: pigging back off the ``O(1)`` speed of arrays by convert data to an array index with a hash function.
* Simplest hash function: modulo.
  * Hash with 5 elements, add 11:
    * ``hash = 11 % 5``
  * Add char ``b``:
    * ``hash = int("b") % 5``
  * Add string ``hi``:
    * ``hash = (int("h") + int("i")) % 5``

## Core: Collisions in Hash Tables

* Linear probing: When you get a collision, just put the element in the next free slot.
  * Could potential result in slower inserts when hash table gets full.
  * Random probing: randomly place element somewhere on collision.
* Separate chaining: keep a list of elements at index locations. Just add elements to list on collision.
  * Has own drawbacks.
* Downsides to hash tables:
  1. Resizing cost:
    * When hash table gets too full (usually 70%), need to resize it.
    * Requires creating a new table and reinserting everything. Potentially very expensive.
  2. Ordering data:
    * Hash tables don't have implicit order in themselves.

## Core: Applications of Hash Tables

* HashSet vs HashMap
  * HashSet doesn't map to a value (standard set).
  * HashMap does.

# Edit Distance

## Core: Overview

* Generate valid words for misspelled words.
* Edit distance of words:
  * Start with ``speel``
  * Close == altered as little as possible.
  * Possible transformations:
    * 1 step away (single character transformation)
      * Substitution (change a single char): ``speel`` -> ``apeel``, ``sbeel``, ``spell``, ``speek``
      * Insertion (add a single char): ``speel`` -> ``bspeel``, ``sipeel``, ``speeel``.
      * Deletion (remove one char): ``speel`` -> ``seel`` -> ``spee``
* Simple spell suggestion algo:
  1. Generate all strings "1 away" from original.
  2. Discard all that aren't words.
* What if not enough? Make it 2 edits away.

## Core: Algorithm

1. Add misspelled words to a queue.
2. While not enough words generated and queue not empty:
  * Remove the a string from the queue.
  * Generate all "1 away" strings from the first string in the queue.
  * Add these to queue.
  * Keep strings that are actual words in a separate list.

## Core: Edit Distance

* What path of words is required to get from ``spell`` to ``mine``?
  * ``spell`` -> ``spill`` -> ``pill`` -> ``pile`` -> ``pine`` -> ``mine``
* Edit distance: number of modifications you need to make to one string to turn into another.
* Solution: build a tree to search problem space.
  * Problem: tree can get extremely large.
    * How many strings are "1 away" from initial word, where k is the length of the word?
        * Substitutions: 25 * k
        * Insertions: 25 * (k + 1) (+1 because you can put an element at the end of the list.
        * Deletions: k
        * Add all together: 52k + 26 new strings
        * Do that for every element of the tree and daaaymn it's a big tree.
* Possible solutions:
  * Dynamic programming -> O(k^2)
  * Pruning: restrict the path to only valid words.
