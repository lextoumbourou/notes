---
title: Transitive Closure
date: 2026-09-26 09:20
modified: 2026-09-26 09:20
status: draft
---

**Transitive Closure** of a [Directed Graphs](directed-graphs.md) G is the digraph G* with the same vertices as G, where G* has a directed edge from u to v whenever G has a directed path from u to v.

In other words, it adds a direct edge for everything that's reachable, so it's a handy way to capture reachability information about a graph. To construct it, you keep adding any missing edge (u, w) where edges (u, v) and (v, w) already exist, until nothing changes.

The same idea applies to a [Relation](relation.md): the transitive closure of a relation is the smallest transitive relation that contains it.

See [Week 13 - Graphs A](../reference/moocs/coursera/uol-discrete-mathematics/week-13-graphs-a.md) and [Week 14 - Graphs B](../reference/moocs/coursera/uol-discrete-mathematics/week-14-graphs-b.md).
