---
title: GPT-6 Luna
slug: gpt-6-luna
date: 2026-09-25 06:50
modified: 2026-09-25 06:50
status: draft
category: model
summary: OpenAI's GPT-6 Luna brings lower token prices, a million-token context window and adjustable reasoning to inexpensive automated workflows.
tags:
  - ModelRelease
  - OpenAI
---

**GPT-6 Luna** is OpenAI's small, inexpensive GPT-6 model, released alongside GPT-6 Sol on **22 September 2026**. The API identifier is `gpt-6-luna` [@openaiAPIChangelog] [@openaiGPT6Luna].

The price is the main attraction for my automated workflows: uncached input is half GPT-5.6 Luna's price, and output drops from $1.20 to $0.50 per million tokens. That makes it a candidate for the many small calls in AI Paper Podcasts, though lower token prices alone do not establish the cost or quality of a complete episode [@openaiAPIPricing].

My [GPT 6 Sol and Luna](gpt-6-sol-and-luna.md) article covers the wider release, developer-reported benchmarks and a hands-on Sol example. This note keeps Luna's API details in one place.

## Availability and reasoning

Specifications checked against OpenAI's documentation on **25 September 2026** [@openaiGPT6Luna]:

| Specification | GPT-6 Luna |
|---|---|
| API identifier | `gpt-6-luna` |
| Input / output | Text and images / text |
| Total context window | 1,050,000 tokens |
| Maximum input | 922,000 tokens |
| Maximum output | 128,000 tokens |
| Knowledge cutoff | 18 May 2026 |
| Reasoning effort | `none`, `low`, `medium` (default), `high`, `xhigh`, `max` |

It supports the Responses and Chat Completions APIs, streaming, structured outputs and function calling. Use **Responses for tools with reasoning enabled**: Chat Completions function calling requires `reasoning_effort="none"` [@openaiGPT6Luna].

## Pricing and Cache Settings

OpenAI Standard API prices, verified **25 September 2026**, in **USD per million tokens**. The first column applies to prompts of at most 272K input tokens; longer prompts use the second column for the entire request [@openaiAPIPricing] [@openaiGPT6Luna].

| Usage | Up to 272K input tokens | Above 272K input tokens |
|---|---:|---:|
| Input, uncached | $0.10 | $0.20 |
| Input, cache read | $0.01 | $0.02 |
| Cache creation/write | $0.125 | $0.25 |
| Output, including reasoning | $0.50 | $0.75 |

Cache reads cost 10% of the uncached input rate; cache writes cost 1.25 times that rate. Batch and Flex are half Standard prices, while Fast mode doubles the applicable rates. A longer reasoning trace can still increase the bill even when the model's token rates are low [@openaiGPT6Luna] [@openaiAPIPricing].
