---
title: "Language Models are Unsupervised Multitask Learners"
date: 2025-07-05 00:00
modified: 2026-09-26 08:55
status: draft
category: reference
---

**Language Models are Unsupervised Multitask Learners** is the 2019 OpenAI paper by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever, which introduced GPT-2.

GPT-2 is a [Transformer](../permanent/transformer.md) [Language Model](../permanent/language-model.md) with up to 1.5 billion parameters, trained on WebText: a dataset of around 8 million web pages linked from Reddit.

The key finding is that a large enough language model, trained only to predict the next word, starts to perform tasks like reading comprehension, summarisation and translation in a zero-shot setting, without any task-specific training.

It follows on from [Improving Language Understanding by Generative Pre-Training](rimproving-language-understanding-by-generative-pre-training.md) (GPT), and was followed by GPT-3 in [Brown et al., 2020: Language Models are Few-Shot Learners](../permanent/brown-et-al-2020-language-models-are-few-shot-learners.md).
