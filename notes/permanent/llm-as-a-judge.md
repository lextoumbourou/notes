---
title: LLM-as-a-Judge
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
tags:
- LLMJudge
- LLMEvals
---

**LLM-as-a-Judge** is an evaluation approach where a [Large Language Models](large-language-models.md) grades the outputs of another model (or itself), instead of a human or a fixed metric. The judge is given the output, usually with the input, a rubric or a reference answer, and asked for a score, a label or a preference between outputs.

It's cheap and scales well, but the judge makes mistakes and has biases of its own, so its scores don't exactly match human judgement. See [How to Correctly Report LLM-as-a-Judge Evaluations](../reference/papers/lee-et-all-2025-how-to-correctly-report-llm-as-a-judge-evaluations.md) and [G-Eval](g-eval.md).
