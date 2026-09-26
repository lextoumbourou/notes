---
title: w2v-BERT
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**w2v-BERT** is a self-supervised speech representation model from Google, introduced by Chung et al. in 2021. It combines two objectives: contrastive learning, as in wav2vec 2.0, and [Masked Language Modelling](masked-language-modelling.md), as in [BERT](bert.md).

Trained on large amounts of unlabelled speech, it learns representations that capture linguistic content, which makes it useful for downstream tasks like speech recognition.

[AudioLM: a Language Modeling Approach to Audio Generation](audiolm-a-language-modeling-approach-to-audio-generation.md) uses w2v-BERT to create its semantic tokens, by running [K-Means](k-means.md) on embeddings from an intermediate layer and using the cluster indices as tokens.
