---
title: Multi-Agent Systems
date: 2024-10-06 00:00
modified: 2024-10-06 00:00
status: hidden
tags:
- AgenticReasoning
---

**Multi-Agent Systems** in [Agentic Reasoning](agentic-reasoning.md) involve multiple models working together to execute [Reflection](reflection.md), [Planning](planning.md) and [Tool Use](tool-use.md) in a collaborative or competitive manner. These agents can communicate with one another, exchange feedback, and coordinate tasks to solve complex problems. Tends to improve upon [Single-Agent Systems](single-agent-systems.md) when collaboration and multiple distinct execution paths are required.

In this system, agents may have specialised roles and communicate with one another to enhance performance, such as through division of labour or consensus-building. While human guidance can still be integrated, the focus is on agents autonomously interacting to optimise the outcomes.

According to [@mastermanLandscapeEmergingAI2024], at the two extremes, there are two main categories of multi-agent system patterns: **Vertical Architecture** and **Horizontal Architectures**.

## Vertical Architectures

In this structure, one agent serves as the leader, with other agents reporting directly to them.

* Depending on the architecture, reporting agents may communicate exclusively with the lead agent.
* Alternatively, a leader may be defined through a shared conversation between all agents.
* The defining features of vertical architectures include having a lead agent and a clear division of labour between the collaborating agents.

## Horizontal Architectures

 In this structure, all agents are treated as equals and participate in a single group discussion about the task.
 
* Communication between agents occurs in a shared thread where each agent can see all messages from the others.
* Agents can also volunteer to complete certain tasks or call tools, meaning they do not need to be assigned by a leading agent.
* Horizontal architectures are generally used for tasks where collaboration, feedback, and group discussion are key to overall success.