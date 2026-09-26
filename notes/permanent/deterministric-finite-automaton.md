---
title: Deterministic Finite Automaton
date: 2023-12-05 00:00
modified: 2026-09-26 08:50
status: draft
---

A **Deterministic Finite Automaton** (DFA) is a type of [Finite Automaton](finite-automaton.md) where each state has exactly one transition for each possible input symbol, so there's only ever one path through the machine for a given input string.

See [Finite Automaton](finite-automaton.md).

Formally, a DFA is a 5-tuple $(Q, \Sigma, \delta, q_0, F)$: a finite set of states $Q$, an input alphabet $\Sigma$, a transition function $\delta: Q \times \Sigma \to Q$, a start state $q_0$ and a set of accept states $F$. It accepts a string if reading the whole string leaves it in an accept state. The languages DFAs recognise are exactly the [Regular Language](regular-language.md), the same as a [Nondeterministic Finite Automaton](nondeterministric-finite-automaton.md).
