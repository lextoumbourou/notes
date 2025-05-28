---
title: Reinforcement Learning from Human Feedback
date: 2025-05-28 00:00
modified: 2025-05-28 00:00
status: draft
aliases:
- RLHF
---

Part of the [Large Language Models](large-language-models.md) training pipeline.

Start with a pre-trained language model, trained on a typically language model objective like next token prediction, or masking objective.

Optionally, can do Supervised Fine-Tuning is about fine-tuning on human-labeled demonstration data. Really just standard supervised learning.

## Reward Model Training

A reward model is trained from human preference data (e.g. pairwise comparisons between model outputs).

## RL Fine-Tuning (e.g. [Proximal Policy Optimization](../../../permanent/proximal-policy-optimization.md) (PPO))

Language Model is fine-tuned using reinforcement learning, with reward model as feedback.

RL Agent samples output pairs from the LLM, scores both using the reward model, and updates the LLM weights accordingly.

Mostly this is done offline, as doing online RLHF would be extremely slow and logistically complex.