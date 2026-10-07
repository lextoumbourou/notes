---
title: Claude Haiku 5.5
slug: claude-haiku-5-5
date: 2026-10-08 06:20
modified: 2026-10-08 06:32
category: model
cover: /_media/claude-haiku-5-5-cover.jpg
cover_credits: Photo by <a href="https://unsplash.com/@zmachacek?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Zdeněk Macháček</a> on <a href="https://unsplash.com/photos/blue-hummingbird-331x7yqD-3k?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
summary: Claude Haiku 5.5 matches GPT-6 Luna's per-token prices for prompts up to 100K tokens, but counts about 1.5 times as many tokens for the same text.
tags:
  - ModelRelease
  - Claude
---

New Haiku model just dropped: Haiku 5.5.

Feels like a lifetime since Anthropic updated Haiku. Turns out Haiku 4.5 came out on 15 October 2025, almost exactly a year ago [@anthropicClaudeHaiku45].

Clearly priced to compete directly with [GPT-6 Luna](gpt-6-luna.md), which is a really great and cheap model that has been my go-to where I need a cost-efficient model.

The per-token costs are roughly on par with Luna, although they get 5x more expensive after 100k tokens, which is interesting.

Some side-by-side comparisons with Luna for specs:

## Specs Haiku vs Luna

| Specification    | Claude Haiku 5.5                 | GPT-6 Luna                           |
| ---------------- | -------------------------------- | ------------------------------------ |
| Released         | 7 October 2026                   | 22 September 2026                    |
| API identifier   | `claude-haiku-5-5`               | `gpt-6-luna`                         |
| Input / output   | Text and images / text           | Text and images / text               |
| Context window   | 1M tokens                        | 1,050,000 tokens (922,000 max input) |
| Maximum output   | 128K tokens                      | 128,000 tokens                       |
| Knowledge cutoff | June 2026                        | 18 May 2026                          |
| Reasoning        | Adaptive thinking, on by default | Reasoning effort, including `none`   |
| Effort levels    | `low` to `max`, default `medium` | `none` to `max`, default `medium`    |

Specs checked against each developer's documentation on **8 October 2026** [@anthropicClaudeHaiku55Docs] [@openaiGPT6Luna] [@openaiAPIChangelog].

## Pricing and Cache Settings

Standard prices in **USD per million tokens**, verified on **8 October 2026**. Costs are different up to and over 100K tokens on Haiku, for some reason. On Luna, the threshold is 272K input tokens. On both, a prompt over the threshold pays the higher rate for the whole request [@anthropicPricing] [@openaiAPIPricing].

| Usage | Haiku 5.5, up to 100K | Haiku 5.5, over 100K | Luna, up to 272K | Luna, over 272K |
|---|---:|---:|---:|---:|
| Input, uncached | $0.10 | $0.50 | $0.10 | $0.20 |
| Input, cache read | $0.01 | $0.05 | $0.01 | $0.02 |
| Cache write | $0.125 (5 min), $0.20 (1 hour) | $0.625 (5 min), $1.00 (1 hour) | $0.125 | $0.25 |
| Output, including reasoning | $0.50 | $2.50 | $0.50 | $0.75 |

Short prompts cost the same per token on both, down to the cache reads and five-minute cache writes. Long prompts are where they split: Haiku's rates go up five times past 100K tokens, while Luna's roughly double, and only past 272K. Both offer a 50% Batch discount. OpenAI also offers Flex at the same discount [@anthropicPricing] [@openaiAPIPricing].

The same launch halved Claude Sonnet 5.5's cache-read price, from $0.20 to $0.10 per million tokens [@anthropicClaudeHaiku55].

## The same text, more tokens

Prices per token only compare models that count tokens the same way. Haiku 5.5 uses the tokeniser introduced with Claude Opus 4.7, which Anthropic says produces about 30% more tokens than Haiku 4.5 for the same text [@anthropicClaudeHaiku55Docs].

I counted six files with each provider's free token-counting endpoint (Anthropic's `count_tokens` and OpenAI's `responses/input_tokens`) on 8 October 2026:

| Sample | Haiku 5.5 tokens | Luna tokens | Ratio |
|---|---:|---:|---:|
| Survey summary (prose) | 9,992 | 5,966 | 1.67 |
| AudioLM summary (prose) | 8,024 | 4,890 | 1.64 |
| Ansible tutorial (prose and commands) | 6,085 | 4,066 | 1.50 |
| [GPT 6 Sol and Luna](gpt-6-sol-and-luna.md) article (prose and code) | 13,847 | 8,946 | 1.55 |
| Python rendering script | 15,708 | 10,037 | 1.56 |
| Generated HTML game | 11,393 | 7,630 | 1.49 |

For my content, Haiku counts **about 1.5 to 1.7 times as many tokens** as Luna. At equal per-token prices, that makes Haiku 5.5 about 1.5 to 1.7 times as expensive for the same input. Output is tokenised the same way, so the same visible answer costs a similar multiple. How many reasoning tokens each model spends on a task is a separate question I haven't measured.

It also means Haiku reaches its 100K threshold sooner than the number suggests. For these samples, 100K Haiku tokens is roughly 60,000 to 67,000 Luna tokens. Between there and Luna's own 272K threshold, the same prompt costs about **7.5 to 8.4 times as much** on Haiku as on Luna. Past 272K Luna tokens (about 405,000 to 455,000 Haiku tokens), Luna's rates double and the gap narrows to roughly 3.7 to 4.2 times.

Anthropic's "75% less than Haiku 4.5" claim already accounts for the newer tokeniser [@anthropicClaudeHaiku55]. It's a comparison with the previous Haiku, not with Luna.

## Benchmarks

Anthropic's announcement compares Haiku 5.5 with GPT-6 Luna. These are developer-reported results; the announcement doesn't state the effort settings for most rows, so this isn't a controlled comparison [@anthropicClaudeHaiku55].

| Benchmark | Haiku 5.5 | GPT-6 Luna | Haiku 4.5 | Sonnet 5.5 |
|---|---:|---:|---:|---:|
| GDPval-AA v2.1, Elo | 1620 | 1437 | 735 | 1840 |
| OSWorld 2.1, offline subset | 72.4% | 48.9% | 15.7% | 83.9% |
| Terminal-Bench 4.0 | 39.2% | 16.4% | 0.0% | 70.6% |
| FrontierCode 1.1, Main | 46.4% | 42.4% | n/a | 52.1% |

If these hold up, Haiku buys more capability per request, while Luna is cheaper for the same text. Which one costs less per *completed task* depends on whether Haiku's results mean fewer retries and fewer calls. I'd need to run my own evals to know.