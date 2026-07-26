---
title: QAGS
date: 2026-07-26 00:00
modified: 2026-07-26 00:00
status: hidden
source: https://arxiv.org/abs/2004.04228
tags:
- NaturalLanguageProcessing
- Benchmark
---

**QAGS** (Question Answering and Generation for Summarization; Wang, Cho & Lewis, 2020) is both a metric and a benchmark for the *factual consistency* of summaries. The metric works on the intuition that if you ask questions about a summary and about its source document, a faithful summary yields the same answers, so divergent answers flag hallucinated content.

As a benchmark it provides human factual-consistency judgments on model-generated summaries for two datasets, CNN/DailyMail and XSUM, so a consistency metric can be scored by how well it correlates with those judgments. This is the hallucination/faithfulness testbed in [Ask, Don’t Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement](../reference/papers/decomposing-llm-judge-scores-into-yes-no-questions.md), where BinEval posts its largest gains, and the factual-consistency counterpart to the broader [SummEval](summeval.md) benchmark.
