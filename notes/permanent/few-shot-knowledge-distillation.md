---
title: Few-Shot Knowledge-Distillation
date: 2025-01-12 00:00
modified: 2025-01-12 00:00
tags:
- LLMPerformance
summary: FSKD routes tasks to cheaper models based on novelty.
---

Good idea from this [blogpost by Steve Krenzel](https://bits.logic.inc/p/getting-gpt-4o-mini-to-perform-like).

A retrieval routing model idea he calls **Few-Shot Knowledge Distillation (FSKD)**.

1. Store input / output task embeddings from a high quality model.
2. For new examples, compute a novelty score: measure how similar to other examples is the new example?
3. Send novel examples to `gpt4o` (or some big, expensive model).
4. Send other examples to `gpt4o-mini` (or some cheaper, lower performance model), but include similar `gpt4o` examples in prompt (few-shot knowledge distillation).

It's cool because it's self-adapting - if the [Domain Shift](domain-shift.md), new examples are sent the larger model until it builds up new examples.

Results on real inventory moderation task:

- 63.4% cost reduction
- Slight improvement over large model: 90.9% accuracy (vs 87.6% for large model alone)
- ~69% of tasks handled by smaller model

![Flowchart example of the FSKD system](../_media/fskd-visualization-pro.svg)