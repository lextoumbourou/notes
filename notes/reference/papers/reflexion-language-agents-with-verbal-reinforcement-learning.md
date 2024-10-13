---
title: "Reflexion: Language Agents with Verbal Reinforcement Learning"
date: 2024-10-04 00:00
modified: 2024-10-04 00:00
status: draft
---

Paper proposes [Reflexion](../../permanent/reflexion.md), a framework for training large language models as [AI Agents](../../permanent/ai-agents.md) that can interact with their environment.

It uses a linguistic feedback mechanism, which allows agents to learn from their mistakes by verbally reflecting on their experiences and storing this information in their memory.

This approach eliminates the need for traditional reinforcement learning techniques that require extensive training data and computationally expensive fine-tuning.

Reflexion broken down into the following parts:
* Actor
    * which generates text and actions
* Evaluator
    * Evaluator model that scores the outputs produced by the Actor.
* Self-Reflection
    * generates verbal reinforcement cues to assist the Actor in self-improvement.
* Memory
    * At inference time, the Actor conditions its decisions on short and long-term memory, similar to the way that humans remember fine-grain recent details while also recalling distilled important experiences from long-term memory.
    * In the RL setup, the trajectory history serves as the short-term memory while outputs from the Self-Reflection model are stored in long-term memory.
    * These two memory components work together to provide context that is specific but also influenced by lessons learned over several trials, which is a key advantage of Reflexion agents over other LLM action choice works.

The paper evaluates Reflexion on various tasks including decision-making, reasoning, and code generation, demonstrating significant performance improvements over baseline approaches.
