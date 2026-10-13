---
title: GPT-6 Sol
slug: gpt-6-sol
date: 2026-10-14 09:05
modified: 2026-10-14 09:05
status: hidden
category: model
summary: OpenAI's mid-sized GPT-6 model for coding and agentic work - half Opus 5.5's token price, a 1M context window and adjustable reasoning.
tags:
  - ModelRelease
  - OpenAI
---

**GPT-6 Sol** is OpenAI's mid-sized GPT-6 model, released alongside GPT-6 Luna on 22 September 2026 [@openaiAPIChangelog]. OpenAI says it's "built to power complex coding and agentic workflows" [@openaiGPT6Sol].

It's half [Claude Opus 5.5](claude-opus-5-5.md)'s price for uncached input and output [@openaiAPIPricing]. OpenAI's docs now point to a newer version, GPT-6.1 Sol, at the same input and output prices [@openaiGPT61Sol].

The wider release, benchmarks and a hands-on example are in [GPT 6 Sol and Luna](gpt-6-sol-and-luna.md). For the small model, see [GPT-6 Luna](gpt-6-luna.md).

## Specs

| Specification | GPT-6 Sol |
|---|---|
| API identifier | `gpt-6-sol` |
| Input / output | Text and images / text |
| Context window | 1,050,000 tokens (922,000 max input) |
| Maximum output | 128K tokens |
| Knowledge cutoff | 20 April 2026 |
| Effort levels | `none` to `max`, default `medium` |

As of 14 October 2026 [@openaiGPT6Sol]. For tool calls with reasoning on, use the Responses API.

## Pricing

USD per million tokens, verified 14 October 2026. Over 272K input tokens, the whole request pays the higher rate [@openaiGPT6Sol].

| Usage | Up to 272K | Over 272K |
|---|---:|---:|
| Input, uncached | $2.00 | $4.00 |
| Input, cache read | $0.20 | $0.40 |
| Cache write | $2.50 | $5.00 |
| Output, including reasoning | $10.00 | $15.00 |

Batch and Flex are half price. Fast mode is double.
