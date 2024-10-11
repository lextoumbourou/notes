---
title: Agentic Reasoning
date: 2024-08-25 00:00
modified: 2024-08-25 00:00
tags:
- MachineLearning
- AgenticReasoning
---

**Agentic Reasoning** refers to an approach to utilising LLMs that involves multi-states of interaction with [AI Agents](ai-agents.md), for example, an agent may plan, reflect on its own output, or even call other tools or agents to reason.

It is the counter to [Zero-Shot Prompting](zero-shot-prompting.md) where the LLM returns the result based on a single prompt.

There are typically 4 high-level categories of activity which constitute Agentic Reasoning, including **Reflection**, **Tool Use**, **Planning** and **Memory**. Additionally, agents can be part of [Single-Agent Systems](single-agent-systems.md), where a single LLM performs Agentic Reasoning, or [Multi-Agent Systems](multi-agent-systems.md) where multiple LLMs work together to achieve a goal.

## Design Patterns
### [Reflection](../../../permanent/reflection.md)

Reflect is the ability of an AI to analyse its own outputs and reasoning.

For example, in paper [Self-Refine: Iterative Refinement with Self-Feedback](../../../reference/self-refine-iterative-refinement-with-self-feedback.md) they describe an approach where an LLM can provide feedback for its own outputs, to improve and refine its approach.

In [Reflexion: Language Agents with Verbal Reinforcement Learning](../../../permanent/reflexion-language-agents-with-verbal-reinforcement-learning.md), they propose an approach to reflection that involves storing mistakes in memory that can be reused.

### [Tool Use](../../../permanent/tool-use.md)

An important pattern in Agentic Reasoning is the ability to leveraging external tools or APIs to gather information or perform actions.

In [Gorilla Large Language Model Connected with Massive APIs](../../../permanent/gorilla-large-language-model-connected-with-massive-apis.md), they fine-tune a model which can perform tasks by retrieving API documents and calling functions. They use test-time modifications to ensure that the model can handle changes to APIs, and it not limited to information in pretrainining.

In [MM-REACT Prompting ChatGPT for Multimodal Reasoning and Action](../../../permanent/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action.md) uses prompting techniques to allow ChatGPT to call vision models and other models to answer questions.

In [SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models](../../../reference/papers-sheet-copilot.md) utilises [Atomic Actions](atomic-actions.md) to represent API commands, which allow the agent to interact with any spreadsheet application.

### [Planning](../../../permanent/planning.md)

Planning in Agentic Reasoning, involves breaking down complex tasks into smaller, manageable steps.  [Open-loop Planning](../../../permanent/open-loop-planning.md) involves creating an entire plan and following it in one go, whereas [Closed-loop Planning](closed-loop-planning.md) involves planning an action, exectuting it, and then planning the next action based on the updated state of the world.

The paper [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../../reference/chain-of-thought-prompting-elicits-reasoning-in-large-language-models.md), is a key example of work in this area which encourages the model to perform a set of "intermediate reasoning steps", the capability for Chain-of-Thought prompting only emerges in Large Language Models, it typically does not help in smaller models.

Another is [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](../../../permanent/hugginggpt-solving-ai-tasks-with-chatgpt-and-its-friends-in-hugging-face.md) which conducts task planning using HuggingFace hosted models.

### [Memory](../../../permanent/memory.md)

### [Single-Agent Systems](single-agent-systems.md)

### [Multi-Agent Systems](multi-agent-systems.md)