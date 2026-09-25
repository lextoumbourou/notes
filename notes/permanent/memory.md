---
title: Memory
date: 2024-10-06 00:00
modified: 2026-09-25 17:10
status: draft
aliases:
- LLM Memory
---

**Memory**, in the context of [Agentic Reasoning](agentic-reasoning.md), refers to [AI Agents](ai-agents.md) storing and loading information outside of their messages and prompts.

An LLM doesn't remember anything between calls; everything it knows about the current task has to fit in its context window. Short-term memory is just that context. Long-term memory is an external store, like files, a database or a vector store, that the agent can write to and retrieve from later, often using [Retrieval Augmented Generation](retrieval-augmented-generation.md).

Papers like [RAISE](raise.md) and [Reflexion](reflexion.md) use memory so agents can remember past actions and outcomes, and do better next time.
