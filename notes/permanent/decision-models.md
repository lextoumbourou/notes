---
title: Decision Models
date: 2026-10-11 09:46
modified: 2026-10-11 09:46
status: hidden
summary: Models that answer predefined questions with typed choices, probabilities or scores.
tags:
- MachineLearning
- AgenticReasoning
---

A **decision model** takes some input and answers questions whose possible outputs are defined in advance. The result might be a choice, a probability for a yes-or-no question, or a score, rather than a free-form response.

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe AI's example. Its three question types are Choice, Score and Noul, a probability for a yes-or-no question [@almeidaIntroducingSystemOne2026] [@typesafePrimitives]. TypeSafe says Jev is trained for calibrated decisions, but has not published enough about its architecture or training method to describe how it works internally [@almeidaIntroducingSystemOne2026].

OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions) offers a similar interface with `choice`, `score` and `predicate` questions, currently using GPT-6 Luna [@openaiDecisionsGuide]. A similar interface does not establish that the underlying models work the same way.

See [Building an agent harness from scratch to pass Ian Goodfellow's intelligence test](building-an-agent-harness-from-scratch-to-pass-ian-goodfellows-intelligence-test.md) for a tool safety example, and [How Jev Works](https://www.youtube.com/watch?v=2j6bs_SAk0s) for my explainer.
