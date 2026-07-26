---
title: Topical-Chat
date: 2026-07-26 00:00
modified: 2026-07-26 00:00
status: hidden
source: https://arxiv.org/abs/2005.00456
tags:
- NaturalLanguageProcessing
- Benchmark
---

**Topical-Chat** started as a knowledge-grounded open-domain dialogue dataset (Gopalakrishnan et al., 2019), where each turn is paired with a "fun fact" the response is meant to build on. As a meta-evaluation benchmark it is used through the human annotations from USR (Mehri & Eskenazi, 2020): 60 dialogue contexts, each answered by six response sources (four decoding methods plus the human and ground-truth replies) for ~360 dialogue-response pairs, rated on dimensions such as naturalness, coherence, engagingness, and groundedness.

Like [SummEval](summeval.md) but for dialogue rather than summarisation, it lets a metric be scored by how well it correlates with those human ratings. This is the dialogue benchmark in [Ask, Don’t Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement](../reference/papers/decomposing-llm-judge-scores-into-yes-no-questions.md), where (following [UniEval](unieval.md)) four of the aspects are used.
