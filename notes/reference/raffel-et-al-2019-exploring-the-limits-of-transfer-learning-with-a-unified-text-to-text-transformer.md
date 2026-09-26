---
title: "Raffel et al., 2019: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
date: 2024-10-13 00:00
modified: 2026-09-26 09:05
status: draft
---

**Raffel et al., 2019** is the paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li and Peter J. Liu.

It introduced **T5** (Text-to-Text Transfer Transformer), which treats every NLP task as a text-to-text problem: the model takes text as input and produces text as output, whether the task is translation, summarisation or classification. That means the same model, loss and training procedure can be used for every task.

The paper is a large study of [Transfer Learning](../permanent/transfer-learning.md) for NLP, comparing pre-training objectives, architectures and datasets. It also introduced the C4 (Colossal Clean Crawled Corpus) dataset.
