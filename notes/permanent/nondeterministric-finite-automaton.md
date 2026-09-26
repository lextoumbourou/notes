---
title: Nondeterministic Finite Automaton
date: 2023-12-05 00:00
modified: 2026-09-26 08:58
status: draft
---

A **Nondeterministic Finite Automaton (NFA)** is a finite automaton where there can be zero, one or several possible transitions for a given state and input, and it can also have empty (ε) transitions that consume no input. It accepts a string if any possible path ends in an accept state.

NFAs recognise exactly the same languages as deterministic finite automata, since any NFA can be converted into an equivalent DFA.

See [Finite Automaton](finite-automaton.md).