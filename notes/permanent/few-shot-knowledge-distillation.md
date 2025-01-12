---
title: Few-Shot Knowledge-Distillation
date: 2025-01-12 00:00
modified: 2025-01-12 00:00
tags:
- LLMPerformance
- LargeLanguageModels
summary: Routes LLMs tasks to cheaper or more powerful models based on task novelty.
---

Good idea from this [blogpost by Steve Krenzel](https://bits.logic.inc/p/getting-gpt-4o-mini-to-perform-like).

A retrieval routing model idea he calls **Few-Shot Knowledge Distillation (FSKD)**.

1. Store input / output task embeddings from a high quality model.
2. For new examples, compute a novelty score: measure how similar to other examples is the new example?
3. Send novel examples to `gpt4o` (or some big, expensive model).
4. Send other examples to `gpt4o-mini` (or some cheaper, lower performance model), but include similar `gpt4o` examples in prompt (few-shot knowledge distillation).

The novelty score is a is determined by:

- The similarity threshold (θ) - which defines how similar an entry needs to be to be considered a match
- The matches threshold (m) - which specifies the number of matches needed to be considered "low novelty"
* And the task embedding (T):

$$
\text{noveltyScore}(T, \theta, m) = \begin{cases}
\text{LOW} & \text{if } |\text{search}(T,\theta)| \geq m \\
\text{HIGH} & \text{otherwise}
\end{cases}
$$

The defaults of $\theta = 0.8$ and m = 3 can be tuned for cost/performance trade-offs.

It's cool because it's self-adapting - if the [Domain Shift](domain-shift.md), new examples are sent the larger model until it builds up new examples.

Results on real inventory moderation task:

- 63.4% cost reduction
- Slight improvement over large model: 90.9% accuracy (vs 87.6% for large model alone)
- ~69% of tasks handled by smaller model

![Flowchart example of the FSKD system](../_media/fskd-visualization-pro.svg)