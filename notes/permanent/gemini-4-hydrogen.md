---
title: Gemini 4 Hydrogen
slug: gemini-4-hydrogen
date: 2026-11-13 06:32
modified: 2026-11-13 06:32
status: draft
category: model
summary: Google's fast everyday coding and agent model, positioned alongside Terra and Sonnet, with a shared 1M-token context window.
tags:
  - ModelRelease
  - Google
---

Google released **Gemini 4 Hydrogen** on 12 November 2026, aimed at everyday coding and agentic work [@kavukcuogluIntroducingGemini4Hydrogen2026].

This looks like Google's answer to Terra and Sonnet. The model card compares it directly with **GPT-6.1 Terra** and **Claude Sonnet 5.5**. Google positions Hydrogen for fast execution, with Carbon and [Gemini 4 Argon](gemini-4-argon.md) handling harder, longer reasoning tasks [@googleDeepMindGemini4HydrogenModelCard].

That includes using Hydrogen as a worker or verifier underneath a more expensive model. Google's new Antigravity Orchestrator switches between Carbon and Hydrogen within a session. Google reports near-Carbon task success on Terminal-Bench 4.0, using 24% less quota and finishing 19% faster on average. Those are results for the combined workflow [@kavukcuogluIntroducingGemini4Hydrogen2026].

## Specs and availability

Google says Hydrogen is adapted from Carbon. The Hydrogen card refers readers to Carbon's card for architecture and training details, without giving a parameter count [@googleDeepMindGemini4HydrogenModelCard].

| Specification | Gemini 4 Hydrogen |
|---|---|
| API identifier | `gemini-4-hydrogen` |
| Inputs | Text, images, PDF, audio and video |
| Output | Text |
| Context window | 1,048,576 tokens, input and output combined |
| Maximum output | 65,536 synchronously; up to 1,048,576 with background long decode, within the shared window |
| Thinking levels | `low`, `medium`, `high` (default), `xhigh` |
| Knowledge cutoff | March 2026 |
| API status | Stable |

Specifications checked on 13 November 2026 [@googleGeminiAPIGemini4Hydrogen] [@googleDeepMindGemini4HydrogenModelCard] [@kavukcuogluIntroducingGemini4Hydrogen2026].

The shared window matters: a million-token input leaves less room for output. Getting responses beyond the synchronous limit requires the long-decode path [@googleGeminiAPIGemini4Hydrogen].

Google announced rollout through AI Studio, the Gemini API, Gemini for business, the Gemini app and Search. Antigravity access is available to paid Google AI Pro and Ultra subscribers [@kavukcuogluIntroducingGemini4Hydrogen2026].

## Benchmarks

Hydrogen leads several knowledge-work and reasoning results in Google's comparison, while Sonnet 5.5 leads the listed terminal-coding and computer-use tests. Selected results from the model card [@googleDeepMindGemini4HydrogenModelCard]:

| Benchmark | Hydrogen | Sonnet 5.5 | GPT-6.1 Terra |
|---|---:|---:|---:|
| SWE-sweep | **13.5%** | 3.1% | 5.8% |
| Terminal-Bench 4.1, cheating-adjusted | 57.0% | **63.9%** | Not reported |
| Agents' Last Exam v1.1 | **71.6%** | 60.4% | 69.8% |
| AutomationBench-AA | **76.1%** | 71.8% | 60.2% |
| OSWorld-2.1, offline subset | 77.6% | **83.9%** | 74.2% |
| HLE-Diamond | **58.8%** | 53.3% | 47.3% |

These are Google's published comparisons, assembled from its own runs and external leaderboards or provider reports. They use different harnesses and settings. Hydrogen generally uses the highest thinking setting, but Terminal-Bench 4.1 uses `high` and averages five attempts per task. OSWorld uses an 82-task offline subset and takes the maximum over three runs. Agents' Last Exam uses the 17 October task version, before its 26 October update [@googleDeepMindGemini4HydrogenEvaluation2026].

## Pricing and cache settings

Standard paid Gemini API prices, in USD per million tokens, checked on 13 November 2026 [@googleGeminiAPIPricing]:

| Usage | Price |
|---|---:|
| Input, uncached | $1.50 |
| Input, cache read | $0.075 |
| Output, including thinking | $7.50 |

Google advertises flat pricing across the full context window. Caching is implicit only: explicit cache creation is unsupported. Batch input and output cost $0.75 and $3.75 respectively per million tokens [@kavukcuogluIntroducingGemini4Hydrogen2026] [@googleGeminiAPIPricing].

## Limitations

The model card warns about hallucinations, occasional slowness or timeouts, and higher token use at higher thinking levels. Google's automated safety evaluations also show a tradeoff: image-to-text safety improves by seven percentage points over Gemini 3.8 Flash, while unjustified refusals increase by 5.57 percentage points [@googleDeepMindGemini4HydrogenModelCard].

Google's targeted frontier-safety testing found Hydrogen below its critical capability levels for biological and cyber risks, while reaching the alert thresholds in both domains. The card says the fuller Safety & Security Report is still forthcoming [@googleDeepMindGemini4HydrogenModelCard].
