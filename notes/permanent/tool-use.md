---
title: Tool Use
date: 2024-10-05 00:00
modified: 2026-09-25 17:19
status: draft
---

**Tool Use** is one of the key design patterns within [Agentic Reasoning](agentic-reasoning.md). It refers to an LLM calling external tools, like a search engine, a calculator, a code interpreter or an API, to do things it can't do with text generation alone.

Typically, the model outputs a structured tool call, the system runs the tool, and the result is fed back into the model's context so it can continue. [ReAct: Synergizing Reasoning and Acting in Language Models](react-synergizing-reasoning-and-acting-in-language-models.md) is an early example of interleaving reasoning with actions like this [@yaoReActSynergizingReasoning2023]. [MCP](mcp.md) is a protocol for connecting models to tools and data sources.

Tool use is the clearest point where LLMs become [AI Agents](ai-agents.md), since they can take actions in the world rather than just returning text.
