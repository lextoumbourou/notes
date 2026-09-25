---
title: Self-Consistency Improves Chain of Thought Reasoning in Language Models
date: 2026-09-25 00:00
modified: 2026-09-25 17:35
status: draft
category: paper
summary: Sample several chains of thought and take a majority vote over the final answers, instead of trusting a single greedy one.
tags:
- LargeLanguageModels
- ReasoningModels
- TestTimeScaling
paper_title: Self-Consistency Improves Chain of Thought Reasoning in Language Models
paper_url: https://arxiv.org/abs/2203.11171
paper_authors: Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou
paper_year: 2022
doi: 10.48550/arXiv.2203.11171
aliases:
- Self-Consistency
---

**Self-Consistency** is a decoding strategy for [Chain-of-Thought Prompting](../../permanent/chain-of-thought-prompting.md), introduced by Google in March 2022, shortly after the original chain-of-thought paper [@wangSelfConsistencyImprovesChain2022].

Instead of greedily decoding a single chain of thought, you sample several diverse reasoning paths from the model, then pick the final answer that comes up most often: effectively a majority vote. The intuition is that a hard problem can usually be solved in several different ways that all lead to the same correct answer, while wrong reasoning tends to lead to different wrong answers.

It needs no extra training, fine-tuning or verifier. It boosted chain-of-thought results on arithmetic and commonsense benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).

The trade-off is compute: you pay for every sampled path. And a majority vote only works when answers can be compared directly, like a number or a multiple-choice option, not open-ended text.

It's one of the earliest examples of [Test-Time Scaling](../../permanent/test-time-scaling.md): spend more compute at inference time to get a better answer. See [LLM Reasoning](../../permanent/llm-reasoning.md) for where it fits in the history.
