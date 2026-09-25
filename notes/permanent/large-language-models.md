---
title: Large Language Models
date: 2024-10-04 00:00
modified: 2026-09-25 17:10
status: draft
aliases:
- LLMs
---

**Large Language Models** or **LLMs** are a type of [Language Model](language-model.md), typically a [Transformer](transformer.md) trained on a next-word prediction task on a massive dataset with a high parameter count.

Training usually happens in stages. First, [Pretraining](pretraining.md) on huge amounts of text produces a base model that's good at continuing text. Then post-training, such as [Supervised Fine-Tuning](supervised-fine-tuning.md) and [Reinforcement Learning from Human Feedback](reinforcement-learning-from-human-feedback.md), turns it into an assistant that follows instructions [@ouyangTrainingLanguageModels2022].

[GPT-3](gpt-3.md) showed that at a large enough scale, models can learn new tasks from just a few examples in the prompt, without any fine-tuning. This is known as [In-Context Learning](in-context-learning.md) [@brownLanguageModelsAre2020a]. More recently, models are also trained to reason before they answer: see [LLM Reasoning](llm-reasoning.md).
