---
title: Planning
date: 2024-09-24 00:00
modified: 2026-09-25 17:10
status: draft
---

**Planning** is a key design pattern of [Agentic Reasoning](agentic-reasoning.md). It refers to an agent breaking a complex task into smaller, manageable steps before or while it carries them out.

Planning can be open-loop, where the agent creates the whole plan up front and executes it without adjustments ([Open-loop Planning](open-loop-planning.md)). Or it can be closed-loop, where the agent plans an action, executes it, observes the outcome, and then plans the next step based on the new state ([Closed-loop Planning](closed-loop-planning.md)).

[Chain-of-Thought Prompting](chain-of-thought-prompting.md) is related, since writing out intermediate steps is a kind of plan. But planning in an agentic system usually means acting on the plan, not just reasoning about it.
