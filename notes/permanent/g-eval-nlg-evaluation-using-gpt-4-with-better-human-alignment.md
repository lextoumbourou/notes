---
title: "G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment"
date: 2026-07-20 00:00
modified: 2026-07-25 00:00
status: hidden
source: https://arxiv.org/abs/2303.16634
tags:
- NaturalLanguageProcessing
- Metrics
---

**G-Eval** is an LLM-as-judge framework that scores generated text with a large model (GPT-4) using a chain-of-thought, form-filling paradigm: it turns the task criteria into evaluation steps, prompts the model to reason through them, then emit a dimension score [@liuGEvalNLGEvaluation2023]. To avoid coarse integer scores it weights the possible scores by the model's output token probabilities, yielding finer-grained continuous ratings that align with human judgment better than [BERTScore](bertscore.md), [MoverScore](moverscore.md), or [BARTScore](bartscore.md), though it inherits LLM-judge weaknesses such as a bias toward text produced by the same model family.

