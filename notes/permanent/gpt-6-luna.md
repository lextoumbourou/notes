---
title: GPT-6 Luna
slug: gpt-6-luna
date: 2026-09-25 06:50
modified: 2026-10-08 07:08
status: hidden
category: model
summary: OpenAI's small GPT-6 model - cheap tokens, a 1M context window and adjustable reasoning.
tags:
  - ModelRelease
  - OpenAI
---

**GPT-6 Luna** is OpenAI's small, cheap GPT-6 model, released alongside GPT-6 Sol on 22 September 2026 [@openaiAPIChangelog].

Input is half GPT-5.6 Luna's price, and output drops from $1.20 to $0.50 per million tokens [@openaiAPIPricing]. It's my go-to when I need a cost-efficient model.

The wider release, benchmarks and a hands-on example are in [GPT 6 Sol and Luna](gpt-6-sol-and-luna.md). For a comparison with Anthropic's equivalent, see [Claude Haiku 5.5](claude-haiku-5-5.md).

## Specs

| Specification | GPT-6 Luna |
|---|---|
| API identifier | `gpt-6-luna` |
| Input / output | Text and images / text |
| Context window | 1,050,000 tokens (922,000 max input) |
| Maximum output | 128K tokens |
| Knowledge cutoff | 18 May 2026 |
| Effort levels | `none` to `max`, default `medium` |

As of 8 October 2026 [@openaiGPT6Luna]. For tool calls with reasoning on, use the Responses API: Chat Completions function calling requires `reasoning_effort="none"`.

## Pricing

USD per million tokens, verified 8 October 2026. Over 272K input tokens, the whole request pays the higher rate [@openaiAPIPricing].

| Usage | Up to 272K | Over 272K |
|---|---:|---:|
| Input, uncached | $0.10 | $0.20 |
| Input, cache read | $0.01 | $0.02 |
| Cache write | $0.125 | $0.25 |
| Output, including reasoning | $0.50 | $0.75 |

Batch and Flex are half price. Fast mode is double.
