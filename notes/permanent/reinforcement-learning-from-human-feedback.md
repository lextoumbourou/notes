---
title: Reinforcement Learning from Human Feedback
date: 2025-05-28 00:00
modified: 2026-09-25 08:34
status: hidden
aliases:
- RLHF
---

**Reinforcement Learning from Human Feedback (RLHF)** is a form of preference tuning used during post-training in the [LLM Training Pipeline](llm-training-pipeline.md) [@raschkaBuildReasoningModel2026].

Start with a pre-trained language model, trained on a language modelling objective like next-token prediction or masked-token prediction.

The model can first undergo [Supervised Fine-Tuning](supervised-fine-tuning.md) on human-labelled demonstration data. InstructGPT used this sequence before reward-model training and RL fine-tuning [@ouyangTrainingLanguageModels2022].

## Reward Model Training

A reward model is trained from human preference data (e.g. pairwise comparisons between model outputs) [@ouyangTrainingLanguageModels2022].

## RL Fine-Tuning (e.g. Proximal Policy Optimization (PPO))

The language model is fine-tuned using reinforcement learning, with the reward model as feedback [@ouyangTrainingLanguageModels2022].

The RL loop samples responses from the LLM, scores them using the reward model, and updates the LLM weights accordingly. Pairwise comparisons are used to train the reward model; the RL loop does not need to sample responses in pairs [@ouyangTrainingLanguageModels2022].

Human feedback can be collected ahead of time. That does not make PPO training offline RL: the policy generates fresh responses during training, and the reward model scores them without needing a human to review each response [@ouyangTrainingLanguageModels2022].
