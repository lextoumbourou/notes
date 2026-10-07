---
title: Claude Haiku 5.5
slug: claude-haiku-5-5
date: 2026-10-08 06:20
modified: 2026-10-08 07:26
category: model
cover: /_media/claude-haiku-5-5-cover.jpg
cover_credits: Photo by <a href="https://unsplash.com/@zmachacek?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Zdeněk Macháček</a> on <a href="https://unsplash.com/photos/blue-hummingbird-331x7yqD-3k?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
summary: New Haiku-class model - this time an actual new Haiku model.
bluesky_post: https://bsky.app/profile/notesbylex.com/post/3mxculxc7ts2e
mastodon_post: https://fedi.notesbylex.com/@lex/117401704030170984
threads_post: https://www.threads.com/@lexisoninsta/post/DeNTh3Jj124
linkedin_post: https://www.linkedin.com/posts/lextoumbourou_claude-haiku-55-share-7513711007286947840-pEBa/
tags:
  - ModelRelease
  - Claude
---

New Haiku model just dropped: Haiku 5.5, released 7 October 2026 [@anthropicClaudeHaiku55].

Feels like a lifetime since Anthropic updated Haiku. Turns out Haiku 4.5 came out on 15 October 2025, almost exactly a year ago - which *is* practically a lifetime in AI terms [@anthropicClaudeHaiku45].

Clearly priced to compete directly with [GPT-6 Luna](gpt-6-luna.md), which is a great, cheap model that's been my go-to when I need a cost-efficient option.

The per-token costs are roughly on par with Luna, although they get 5x more expensive after 100k tokens, which is interesting.

## Specs Haiku vs Luna

| Specification    | Claude Haiku 5.5                 | GPT-6 Luna                           |
| ---------------- | -------------------------------- | ------------------------------------ |
| API identifier   | `claude-haiku-5-5`               | `gpt-6-luna`                         |
| Input / output   | Text and images / text           | Text and images / text               |
| Context window   | 1M tokens                        | 1,050,000 tokens (922,000 max input) |
| Maximum output   | 128K tokens                      | 128K tokens                       |
| Knowledge cutoff | June 2026                        | 18 May 2026                          |
| Reasoning        | Adaptive thinking, on by default | Reasoning effort, including `none`   |
| Effort levels    | `low` to `max`, default `medium` | `none` to `max`, default `medium`    |

As of 8 October 2026 [@anthropicClaudeHaiku55Docs] [@openaiGPT6Luna].

## Pricing and Cache Settings

Haiku's price per token goes up 5x over 100K. Luna also has an increase for larger context sizes, but only 2x/1.5x input/output, and its threshold is 272K input tokens. On both, a prompt over the threshold pays the higher rate for the whole request [@anthropicPricing] [@openaiAPIPricing].

Prices in USD per million tokens, as of 8 October 2026.

| Usage | Haiku 5.5, up to 100K | Haiku 5.5, over 100K | Luna, up to 272K | Luna, over 272K |
|---|---:|---:|---:|---:|
| Input, uncached | $0.10 | $0.50 | $0.10 | $0.20 |
| Input, cache read | $0.01 | $0.05 | $0.01 | $0.02 |
| Cache write | $0.125 (5 min), $0.20 (1 hour) | $0.625 (5 min), $1.00 (1 hour) | $0.125 | $0.25 |
| Output, including reasoning | $0.50 | $2.50 | $0.50 | $0.75 |

Also, of note, the same launch halved Claude Sonnet 5.5's cache-read price, from $0.20 to $0.10 per million tokens [@anthropicClaudeHaiku55].

## Tokenisers

Haiku 5.5 uses the tokeniser introduced with Claude Opus 4.7, which tends to produce about 30% more tokens than Haiku 4.5 (and other models before Opus 4.7) for the same text [@anthropicClaudeHaiku55Docs].

I counted six files with each provider's free token-counting endpoint (Anthropic's `count_tokens` and OpenAI's `responses/input_tokens`) on 8 October 2026:

| Sample | Haiku 5.5 tokens | Luna tokens | Ratio |
|---|---:|---:|---:|
| Survey summary (prose) | 9,992 | 5,966 | 1.67 |
| AudioLM summary (prose) | 8,024 | 4,890 | 1.64 |
| Ansible tutorial (prose and commands) | 6,085 | 4,066 | 1.50 |
| [GPT 6 Sol and Luna](gpt-6-sol-and-luna.md) article (prose and code) | 13,847 | 8,946 | 1.55 |
| Python rendering script | 15,708 | 10,037 | 1.56 |
| Generated HTML game | 11,393 | 7,630 | 1.49 |

So about 1.5 to 1.7 times as many tokens as Luna, which means Haiku costs about 1.5 to 1.7 times as much for the same text.

## Benchmarks

Anthropic's announcement compares Haiku 5.5 with GPT-6 Luna, and as you might expect, they report a few improvements over Luna, some of them pushing towards Sonnet 5.5 quality [@anthropicClaudeHaiku55].

| Benchmark | Haiku 5.5 | GPT-6 Luna | Haiku 4.5 | Sonnet 5.5 |
|---|---:|---:|---:|---:|
| GDPval-AA v2.1, Elo | 1620 | 1437 | 735 | 1840 |
| OSWorld 2.1, offline subset | 72.4% | 48.9% | 15.7% | 83.9% |
| Terminal-Bench 4.0 | 39.2% | 16.4% | 0.0% | 70.6% |
| FrontierCode 1.1, Main | 46.4% | 42.4% | n/a | 52.1% |

Overall, Luna is still the cheaper model, but potentially the performance gains might be worth the extra token cost.

