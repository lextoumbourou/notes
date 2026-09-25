---
title: Reflection
date: 2024-09-29 00:00
modified: 2026-09-25 17:10
status: draft
---

**Reflection** is a design pattern in [Agentic Reasoning](agentic-reasoning.md) where a model reviews its own output, critiques it, and uses that feedback to improve the next attempt.

[Self-Refine](self-refine.md) does this in a loop with a single LLM: generate an answer, give feedback on it, refine it, and repeat. [Reflexion](reflexion.md) stores verbal reflections on past failures in memory, so the agent can do better on later attempts.

Reflection can help, but it's only as good as the model's ability to spot its own mistakes.
