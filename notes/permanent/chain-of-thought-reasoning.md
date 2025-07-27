---
title: Chain-of-Thought Reasoning
date: 2025-07-28 00:00
modified: 2025-07-28 00:00
status: draft
tags:
- ReasoningModels
aliases:
- Chain-of-Thought
- CoT
---

**Chain-of-Thought Reasoning** is an LLM Reasoning technique where the model can reasoning in token space. Originally described as a prompting technique [Chain-of-Thought Prompting](chain-of-thought-prompting.md) where the model was given few-shot examples of input / output  examples with intermediary reasoning but later with the introduction of models like OpenAI's o1 and [DeepSeek-R1-Zero](DeepSeek-R1-Zero.md), allowed the models to perform reasoning without few-shot examples, either by learning to reasoning as a fine-tuning step (by adding reasoning steps into training data) or via reinforcement learning, where the model was rewarded for applying a thinking process before returning an output. 