---
title: Spanning Tree
date: 2025-02-16 00:00
modified: 2025-02-16 00:00
aliases:
- "Spanning Trees"
summary: "a sub graph of a connected graph that contains all vertices, but no cycles"
tags:
- ComputerScience
- GraphTheory
---

The **Spanning Tree** of a connected graph $G$ is a connected sub graph which contains all vertices of $G$, but with no cycles.

## Example

$G$ is a connected graph, $T_1$, $T_2$, $T_3$ and $T_4$ are spanning trees.

![week-15-spanning-tree.webp](../_media/week-15-spanning-tree.webp)

## Minimum Spanning Tree (MST)

The Minimum Spanning Tree (MST) is the lowest cost spanning tree within the graph, where vertices have associated costs.

Two algorithms for computing the MST:

* [Prim's Algorithm](prims-algorithm.md)
* [Kruskal's Algorithm](kruskals-algorithm.md).
