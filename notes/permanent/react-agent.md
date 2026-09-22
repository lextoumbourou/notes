---
title: ReAct
date: 2024-10-06 00:00
modified: 2024-10-06 00:00
status: draft
summary: A simple agent loop that interleaves reasoning, tool actions and observations.
tags:
- AgenticReasoning
- LanguageModels
---

[ReAct: Synergizing Reasoning and Acting in Language Models](react-synergizing-reasoning-and-acting-in-language-models.md) is a simple agent loop that interleaves reasoning with actions. Instead of asking a language model to reason once and then produce an answer, the harness lets it repeatedly decide what to do, act on the environment, and use the result to decide what to do next.

The loop is:

1. **Reason:** decide what information or action is needed.
2. **Act:** call a tool or interact with the environment.
3. **Observe:** receive the result and add it to the context.

The model repeats this cycle until it has enough information to finish. The reasoning traces help it track a plan and respond to unexpected results, while actions provide information that is not available in the model's initial context. The original [ReAct: Synergizing Reasoning and Acting in Language Models](react-synergizing-reasoning-and-acting-in-language-models.md) demonstrated this pattern for question answering, fact verification and interactive decision-making tasks.
