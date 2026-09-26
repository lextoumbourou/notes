---
title: "Improving Language Understanding by Generative Pre-Training"
date: 2024-10-13 00:00
modified: 2026-09-26 08:55
status: draft
---

**Improving Language Understanding by Generative Pre-Training** is the 2018 OpenAI paper by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever, which introduced [GPT](../permanent/gpt.md).

The approach has two stages:

1. Generative pre-training of a [Transformer](../permanent/transformer.md) decoder as a [Language Model](../permanent/language-model.md) on a large corpus of unlabelled text (BooksCorpus).
2. Discriminative [Fine-Tuning](../permanent/fine-tuning.md) on each labelled downstream task, with minimal changes to the model architecture.

This approach improved the state of the art on 9 of the 12 tasks studied. It was followed by GPT-2 in [Language Models are Unsupervised Multitask Learners](language-models-are-unsupervised-multitask-learners.md).
