---
title: Dependency Grammar
date: 2026-01-06 00:00
modified: 2026-01-06 00:00
status: draft
---

**Dependency Grammar** is another type of grammar for expressing how to create valid [Syntax](syntax.md).

Unlike [Context-Free Grammar](context-free-grammar.md), they operate over words, not phrases.

It's a labled directed graph.

A **dependnency arc** is a directed link between two words in a sentence that encodes a syntactic relationship. It has 3 parts:

HEAD - (relation) -> DEPENDENT

Consider the sentence:

> The black cat chased the mouse

Some arcs in the dependecy tree are:

* `chased -> nsubj -> cat`
* `chased -> obj -> mouse`
* `cat -> det -> the`
* `cat -> amod -> black`

Each arc states:
- which word is the head
- which word depends on it
- how it depends (the label)

In a [Constituents](constituency.md) grammar, structure is expressed via phrases.

But, in dependency grammy, the same information is encoded via arcs.

The head of a dependency grammar is usually a "tensed verb"

> The cat **chased** the mouse.

Root:

chased (ROOT)

All major arguemnts and modiefi depend (direclty or indirectly) on it.

We say a dependency tree is **Projective** is not depency arcs cross when words are laid out in linear order - in other words, it has no crossing dependencies

![image-53.png](image-53.png)
