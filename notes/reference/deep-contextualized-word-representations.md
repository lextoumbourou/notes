---
title: "Deep Contextualized Word Representations"
date: 2024-10-13 00:00
modified: 2026-09-26 08:58
status: draft
aliases:
- Peters et al., 2018
---

This paper introduces a new word representation called [ELMo](../permanent/elmo.md), which leverages a deep, bidirectional language model (biLM) to create context-sensitive word vectors.

The authors demonstrate that these representations significantly improve performance on various challenging natural language processing (NLP) tasks such as question answering and sentiment analysis.

ELMo effectively captures both semantic and syntactic information about words within their context, which traditional word embeddings fail to achieve.

The authors analyse ELMo's performance on various tasks and explore the different types of information encoded in different layers of the biLM, highlighting the benefits of using all layers for optimal performance.

Finally, they demonstrate that ELMo improves sample efficiency, reducing the amount of training data required to achieve high performance.