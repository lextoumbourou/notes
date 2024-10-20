---
title: Multi-Agent Systems
date: 2024-10-06 00:00
modified: 2024-10-06 00:00
status: draft
tags:
- AgenticReasoning
---

**Multi-Agent Systems** in [Agentic Reasoning](agentic-reasoning.md) involve multiple models working together to execute [Reflection](reflection.md), [Planning](planning.md), and [Tool Use](tool-use.md) in a collaborative or competitive manner. These agents can communicate with one another, exchange feedback, and coordinate tasks to solve complex problems. Tends to improve upon [Single-Agent Systems](single-agent-systems.md) more when collaboration and multiple distinct execution paths are required [^1].

In this system, agents may have specialised roles and communicate with one another to enhance performance, such as through division of labor or consensus-building. While human guidance can still be integrated, the focus is on agents autonomously interacting to optimise the outcomes.

According to [^1], at the two extremes, there are two main categories of multi-agent system pattern.

## [Vertical Architectures](Vertical%20Architectures)

 In this structure, one agent acts as a leader and has other agents report directly to them.
    * Depending on the architecture, reporting agents may communicate exclusively with the lead agent.
    * Alternatively, a leader may be defined with a shared conversation between all agents.
    * The defining features of vertical architectures include having a lead agent and a clear division of labor between the collaborating agents.

## [Horizontal Architectures](Horizontal%20Architectures)

 In this structure, all the agents are treated as equals and are part of one group discussion about the task.
    * Communication between agents occurs in a shared thread where each agent can see all messages from the others.
    * Agents also can volunteer to complete certain tasks or call tools, meaning they do not need to be assigned by a leading agent.
    * Horizontal architectures are generally used for tasks where collaboration, feedback and group discussion are key to the overall success of the task.

## Key Papers

[The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey](../../../permanent/the-landscape-of-emerging-ai-agent-architectures-for-reasoning-planning-and-tool-calling-a-survey.md)
* Surveys existing single and multi-agent architectures, and defines the **Vertical** and **Horizontal** architectural patterns.

[^1]: Masterman, T., Besen, S., Sawtell, M., & Chao, A. (2024). The landscape of emerging AI agent architectures for reasoning, planning, and tool calling: A survey. arXiv. https://arxiv.org/abs/2404.11584
