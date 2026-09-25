---
title: Latent Reasoning
date: 2025-07-28 00:00
modified: 2026-09-25 17:10
status: draft
tags:
- LatentReasoning
- ReasoningModels
---

**Latent Reasoning** is an [LLM Reasoning](llm-reasoning.md) paradigm where the model performs computations within its internal hidden space, as opposed to reasoning at the token level in [Chain-of-Thought Reasoning](chain-of-thought-reasoning.md).

One example is Coconut (Chain of Continuous Thought). Instead of decoding the last hidden state into a token, the model feeds it straight back in as the next input embedding, so it can reason without being constrained to words [@haoTrainingLargeLanguage2026].

It's worth mentioning that the major distinguishing factor between the [total doom scenario](https://ai-2027.com/race) and [not](https://ai-2027.com/slowdown) in [AI 2027](https://ai-2027.com/) is our ability to interpret the reasoning of AI models. So I guess, as with everything in AI in 2025, we'll keep exploring this paradigm at our own risk.
