---
title: SummEval
date: 2026-07-25 00:00
modified: 2026-07-25 00:00
status: hidden
source: https://arxiv.org/abs/2007.12626
tags:
- NaturalLanguageProcessing
- Benchmark
---

**SummEval** is a benchmark for *evaluating summarisation evaluation*: rather than scoring summaries directly, it measures how well automatic metrics agree with human judgment. Fabbri et al. (2021) collected summaries from 23 summarisation models over 100 CNN/DailyMail news articles, then gathered ratings from both expert and crowd annotators on four dimensions, coherence, consistency, fluency, and relevance, on a 1-5 Likert scale.

Alongside the annotations they released a unified toolkit reimplementing 14 automatic metrics, so a new metric can be meta-evaluated by correlating its scores against the human ratings. This is why it is the standard testbed in papers like [Ask, Don’t Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement](../reference/papers/decomposing-llm-judge-scores-into-yes-no-questions.md), where BinEval, [G-Eval](g-eval.md), and [UniEval](unieval.md) are compared on their per-dimension correlation with human scores.
