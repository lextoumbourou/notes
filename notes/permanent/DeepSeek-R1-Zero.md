---
title: DeepSeek-R1-Zero
date: 2025-01-21 00:00
modified: 2026-09-25 17:10
status: draft
tags:
- ReasoningModels
---

**DeepSeek-R1-Zero** is a reasoning model from the [DeepSeek-R1 Reasoning via Reinforcement Learning](../reference/papers/deepseek-r1-reasoning-via-reinforcement-learning.md) paper, trained to reason without explicitly seeing reasoning traces.

DeepSeek applied [Reinforcement Learning (RL)](reinforcement-learning.md) directly to the DeepSeek-V3-Base model, with no supervised fine-tuning first. The model was only rewarded for getting the correct answer and for putting its reasoning in the required format. The reasoning behaviour emerged on its own [@deepseek-aiDeepSeekR1IncentivizingReasoning2025].

It went from 15.6% to 71.0% on AIME 2024 during training, and learned to write longer chains of thought and check its own work. But its outputs were hard to read and often mixed languages, which is why DeepSeek went on to build DeepSeek-R1 with a small amount of supervised data first.
