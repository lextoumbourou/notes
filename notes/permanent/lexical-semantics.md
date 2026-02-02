---
title: Lexical Semantics
date: 2025-11-16 00:00
modified: 2026-02-02 00:00
status: draft
tags:
- NaturalLanguageProcessing
---

Lexical Semantics is the branch of linguistics and NLP that studies how words encode meaning and relate to each other.

## Hypernyms and Hyponyms

These describe the hierarchical "is-a" relationship in lexical semantics.

Hypernym refers to the more general - or superordinate - term: an "animal" is a hypernym of dog. Think "hyper" -> over/above like *hyper*active, for higher, broader category.

Hyponym is the more specific (subordinate) term: "dog" is a hyponym of animal.

Think "hypo" -> under/below like *hypo*thermia for the lower, specific term.

In [RDFS](rdfs.md) the `rdfs:subClassOf` predicate captures the hyponym -> hypernym relationship between terms.

## Hyponymy

The relationship between hypernyms and hyponyms is called "**hyponymy**": e.g., "dog is-a animal".