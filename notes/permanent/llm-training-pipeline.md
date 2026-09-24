---
title: LLM Training Pipeline
date: 2026-04-29 00:00
modified: 2026-09-25 08:44
summary: "The two steps to training modern LLMs."
tags:
- LargeLanguageModels
- MachineLearning
aliases:
- LLM Training Recipe
cover: /_media/llm-training-pipeline-taylor-vick.jpg
cover_credits: Photo by <a href="https://unsplash.com/@tvick">Taylor Vick</a> on <a href="https://unsplash.com/photos/cable-network-M5tzZtFCOfs">Unsplash</a>
---

[Large Language Models](large-language-models.md) training is typically structured into two phases: pre-training and post-training [@raschkaBuildReasoningModel2026].

## Pre-training

In the [Pre-training](pretraining.md) step, LLMs are trained on giant collections of data, often gathered from the internet, books and other sources. These collections can reach terabytes or petabytes, depending on the training run. Autoregressive LLMs are typically trained with the next-token-prediction objective. To get good at doing that, the model needs to learn patterns in language and the relationships represented in the dataset, developing a sort of "understanding" of it. Large pre-training runs can take weeks to months and are typically extremely expensive [@raschkaBuildReasoningModel2026].

## Post-training

Once we have a pre-trained LLM, we fine-tune it for a task, like responding to chat messages. Techniques include [Supervised Fine-Tuning](supervised-fine-tuning.md), including instruction tuning and preference tuning to teach it to respond to user queries in ways people prefer. Instruction tuning is a form of supervised fine-tuning, rather than a name for all supervised fine-tuning [@raschkaBuildReasoningModel2026] [@ouyangTrainingLanguageModels2022].

[Reinforcement Learning from Human Feedback](reinforcement-learning-from-human-feedback.md) (RLHF) is a form of preference tuning. Direct Preference Optimisation (DPO) is another approach: it learns directly from preferred and rejected responses without a separate reward-model training stage or an RL rollout loop [@ouyangTrainingLanguageModels2022] [@rafailovDirectPreferenceOptimization2023].