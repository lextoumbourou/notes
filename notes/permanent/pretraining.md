---
title: Pretraining
date: 2026-04-29 00:00
modified: 2026-09-25 08:34
status: hidden
aliases:
- Pre-training
---

**Pre-training** is the first phase of the typical [LLM Training Pipeline](llm-training-pipeline.md). LLMs are trained on giant collections of data from the internet, books and other sources. Autoregressive LLMs are typically trained with the next-token-prediction objective, learning language patterns and relationships represented in the dataset [@raschkaBuildReasoningModel2026].

Large pre-training runs can take weeks to months and are typically extremely expensive. The result is a pre-trained, or base, model, which can then be adapted through post-training, including [Supervised Fine-Tuning](supervised-fine-tuning.md) and preference tuning [@raschkaBuildReasoningModel2026].
