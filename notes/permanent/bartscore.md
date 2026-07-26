---
title: BARTScore
date: 2026-07-25 00:00
modified: 2026-07-25 00:00
status: hidden
source: https://arxiv.org/abs/2106.11520
tags:
- NaturalLanguageProcessing
- Metrics
---

**BARTScore** reframes evaluation as a text-generation problem: it feeds one text into a pretrained sequence-to-sequence model (BART) and scores the other as the model's log-likelihood of generating it, so more probable output scores higher [@yuanBARTScoreEvaluatingGenerated2021]. Unlike [BERTScore](bertscore.md) and [MoverScore](moverscore.md), which match embeddings, this is a model-scoring approach, and changing the conditioning direction (source→output, output→reference, reference→output) lets the same metric target faithfulness, precision, or recall rather than a single overlap number.
