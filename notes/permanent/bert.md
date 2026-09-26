---
title: BERT
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**BERT** (Bidirectional Encoder Representations from Transformers) is an encoder-only [Transformer](transformer.md) language model released by Google in 2018. It was introduced in the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../reference/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding.md).

BERT is pre-trained on unlabelled text with [Masked Language Modelling](masked-language-modelling.md), where some tokens are hidden and the model predicts them from the context on both sides, plus a next sentence prediction task. The pre-trained model can then be fine-tuned for downstream tasks like classification or question answering.

It inspired many follow-ups, including [RoBERTa](roberta.md), and the masked prediction idea was carried over to speech in models like [HuBERT](hubert.md) and [w2v-BERT](w2v-bert.md).
