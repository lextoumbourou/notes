---
title: "StreamingLLM"
date: 2026-09-22 07:15
modified: 2026-09-22 07:15
status: draft
summary: "A method for streaming language-model inference with a bounded KV cache."
source: https://arxiv.org/abs/2309.17453v4
tags:
- ContextManagement
- LargeLanguageModels
---

**StreamingLLM** retains the KV states of initial attention-sink tokens and recent tokens while evicting older states. This supports continued streaming inference with a bounded cache; it does not preserve access to the entire earlier conversation.

Source: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453v4).

Referenced in [An Empirical Study of Harness Design for Coding Agents](an-empirical-study-of-harness-design-for-coding-agents.md).
