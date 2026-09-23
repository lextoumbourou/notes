---
title: Chain-of-Thought Reasoning
date: 2025-07-28 00:00
modified: 2026-09-24 07:34
status: hidden
tags:
- ReasoningModels
aliases:
- Chain-of-Thought
- CoT
---

**Chain-of-Thought Reasoning** is an [LLM Reasoning](llm-reasoning.md) technique where an LLM can reason in token space: in other words, it generates reasoning traces as text before generating an answer.

It can be elicited through [Chain-of-Thought Prompting](chain-of-thought-prompting.md), where the model is given examples with intermediate reasoning, or through [Think Step-by-Step](think-step-by-step.md) prompting without examples. Models like OpenAI's o1 and [DeepSeek-R1-Zero](DeepSeek-R1-Zero.md) were subsequently trained to reason using reinforcement learning. For R1-Zero, the rewards evaluated answer correctness and output format, rather than prescribing each reasoning step.

See [LLM Reasoning](llm-reasoning.md) for the history and its relationship to [Agentic Reasoning](agentic-reasoning.md), human cognition and [Logical Reasoning](logical-reasoning.md).
