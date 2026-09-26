---
title: "Howard & Ruder, 2018: Universal Language Model Fine-tuning for Text Classification"
date: 2024-10-13 00:00
modified: 2026-09-26 08:55
status: draft
---

**Universal Language Model Fine-tuning for Text Classification** is a 2018 paper by Jeremy Howard and Sebastian Ruder ([arXiv](https://arxiv.org/abs/1801.06146)), which introduced ULMFiT, a method for [Transfer Learning](../permanent/transfer-learning.md) in NLP.

ULMFiT has three stages:

1. Pre-train an LSTM [Language Model](../permanent/language-model.md) (AWD-LSTM) on a large general corpus (Wikitext-103).
2. [Fine-Tuning](../permanent/fine-tuning.md) the language model on text from the target task.
3. Fine-tune a classifier on top of the language model.

It introduced techniques like discriminative fine-tuning, slanted triangular learning rates and gradual unfreezing, to avoid forgetting what was learned during pre-training.

It showed that transfer learning, which was already standard in computer vision, works well for NLP too. ULMFiT is covered in [Deep Learning for Coders with Fastai and Pytorch: AI Applications Without a PhD](books/deep-learning-for-coders-with-fastai-and-pytorch.md).
