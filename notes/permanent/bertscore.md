---
title: BERTScore
date: 2026-07-25 00:00
modified: 2026-07-25 00:00
status: hidden
source: https://arxiv.org/abs/1904.09675
tags:
- NaturalLanguageProcessing
- Metrics
---

**BERTScore** scores a candidate against a reference by matching their contextual BERT token embeddings, greedily pairing each token with its most similar counterpart and aggregating the cosine similarities into precision, recall, and F1 [@zhangBERTScoreEvaluatingText2019]. Unlike [ROUGE: A Package for Automatic Evaluation of Summaries](../../../reference/papers/rouge-a-package-for-automatic-evaluation-of-summaries.md)'s exact word overlap, this captures semantic and paraphrase similarity, so "a quick fox" and "a fast fox" score highly, but the result is a single opaque similarity number that says nothing about factuality or which content is missing.
