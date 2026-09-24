---
title: Supervised Fine-Tuning
date: 2026-09-25 08:34
modified: 2026-09-25 08:38
status: hidden
summary: Adapting a pre-trained model using examples of the desired responses.
tags:
  - LargeLanguageModels
  - MachineLearning
aliases:
  - SFT
---

**Supervised Fine-Tuning (SFT)** is about [Fine-Tuning](fine-tuning.md) a pre-trained model on labelled demonstration data. Really just standard supervised learning, continuing from the pre-trained weights [@ouyangTrainingLanguageModels2022].

In the [LLM Training Pipeline](llm-training-pipeline.md), this can teach the model to respond to user queries using examples of instructions and their desired responses. This form of SFT is called **instruction tuning**. SFT is the broader term, since supervised fine-tuning can also target other tasks [@raschkaBuildReasoningModel2026].

Preference tuning, such as [RLHF](reinforcement-learning-from-human-feedback.md) or DPO, can follow SFT. Instead of only learning to imitate demonstrations, it uses feedback about which responses are preferred [@ouyangTrainingLanguageModels2022] [@rafailovDirectPreferenceOptimization2023].
