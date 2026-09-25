---
title: Test-Time Scaling
category: note
date: 2026-06-13 00:00
modified: 2026-09-25 17:10
status: hidden
tags:
- LargeLanguageModels
- TestTimeScaling
- ReasoningModels
---

**Test-Time Scaling** is a series of techniques that improve a model's answer by spending more compute at inference time, rather than by training a bigger model. Typically it refers to improving a model's answer by spending more reasoning/thinking tokens before responding.

It was popularised by OpenAI's o1 models, which were trained to reason before answering and let developers control how much reasoning effort to spend [@openaiO1Model]. See [LLM Reasoning](llm-reasoning.md).

There are two broad approaches:

- **Sequential**: the model thinks for longer, generating a longer chain of thought before it answers.
- **Parallel**: the model generates several answers and one is picked, either by majority vote, as in self-consistency [@wangSelfConsistencyImprovesChain2022], or by a verifier that scores each candidate.

Parallel approaches only help if you can reliably pick the right answer. [Heavy Thinking: A Test-Time Scaling Pattern for Hard Problems](../reference/papers/heavythinking-a-test-time-scaling-pattern-for-hard-problems.md) combines the two: subagents reason in parallel, then another LLM deliberates over their answers sequentially.
