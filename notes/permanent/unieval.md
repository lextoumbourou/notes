---
title: UniEval
date: 2026-07-25 00:00
modified: 2026-07-25 00:00
status: hidden
source: https://arxiv.org/abs/2210.07197
tags:
- NaturalLanguageProcessing
- Metrics
---

**UniEval** reframes text-generation evaluation as a Boolean question-answering task: a single T5-based model is asked **one** yes/no question per quality dimension (e.g. "Is this a coherent summary?"), and the probability it assigns to "Yes" becomes that dimension's score [@zhongUnifiedMultiDimensionalEvaluator2022]. Because each dimension is just a natural-language question, one unified model covers coherence, consistency, fluency and relevance, and it can extend to new dimensions by adding a question. This one-question-per-dimension design is the direct precursor to [Ask, Don’t Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement](../reference/papers/decomposing-llm-judge-scores-into-yes-no-questions.md)'s BinEval, which keeps the binary-question idea but decomposes each dimension into **many** questions rather than one.
