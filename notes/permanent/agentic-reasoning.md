---
title: Agentic Reasoning
date: 2024-08-25 00:00
modified: 2024-08-25 00:00
summary: an approach to utilising LLMs that involves multi-states of interaction
cover: /_media/agentic-reasoning-overview.png
hide_cover_in_article: true
tags:
- MachineLearning
- AgenticReasoning
---

**Agentic Reasoning** is about approaches to LLMs that involve multi-state interaction, as opposed to [Zero-Shot Prompting](zero-shot-prompting.md), where the LLM returns the result based on a single prompt.

We think of LLM interactions as [AI Agents](ai-agents.md); agents can reflect, plan, use tools and store things in long-term memory. Additionally, agents can work collaboratively with other agents in [Multi-Agent Systems](multi-agent-systems.md).

![Agentic Reasoning Workflow Overview](../_media/agentic-reasoning-overview.png)

## Design Patterns

### [Reflection](reflection.md)

Reflection is the ability of an agent to analyse its outputs and prior reasoning. For example, [Self-Refine](self-refine.md) is an approach where an LLM can provide feedback for its outputs to improve and refine its strategy in iterations. [Reflexion](reflexion.md) is an alternative approach that involves storing mistakes in memory that can be reused.

### [Planning](planning.md)

Planning involves breaking down complex tasks into smaller, manageable steps. [Chain-of-Thought Prompting](chain-of-thought-prompting.md) is a key example of work in this area, which encourages the model to perform a set of "intermediate reasoning steps", which significantly improves its ability to reason through complex tasks.

Some systems use [Open-loop Planning](../../../permanent/open-loop-planning.md), which involves creating an entire plan and following it in one go, whereas [Closed-loop Planning](closed-loop-planning.md) consists of planning an action, executing it, and then planning the next action based on the updated state of the world.

### [Tool Use](../../../permanent/tool-use.md)

Using tools is key, as agents can leverage external tools or APIs to gather information or perform actions, another important component of an agentic system.

In [Gorilla Large Language Model Connected with Massive APIs](../../../permanent/gorilla-large-language-model-connected-with-massive-apis.md), they fine-tune a model that can perform tasks by retrieving API documents and calling functions. They use test-time modifications to ensure that the model can handle changes to APIs and is not limited to information in pre-training.

In [MM-REACT Prompting ChatGPT for Multimodal Reasoning and Action](../../../permanent/mm-react-prompting-chatgpt-for-multimodal-reasoning-and-action.md) uses prompting techniques to allow ChatGPT to call vision models and other models to answer questions.

In [SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models](../../../reference/papers-sheet-copilot.md) utilises [Atomic Actions](atomic-actions.md) to represent API commands, which allow the agent to interact with any spreadsheet application.

More recent papers like [ToolGen: Unified Tool Retrieval and Calling via Generation](../../../permanent/toolgen-unified-tool-retrieval-and-calling-via-generation.md), investigating including tool calls are part of the token vocabulary.

### [Memory](memory.md)

Recent papers have introduced a paradigm of long-term memory, such as [RAISE](raise.md) and [Reflexion](reflexion.md).

### Act / Observe

In some systems, the model will take action and then observe the results; in some examples, the agent can update its plan based on feedback from the environment.
