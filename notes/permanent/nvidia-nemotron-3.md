---
title: NVIDIA Nemotron 3
date: 2025-12-15 00:00
modified: 2026-09-24 07:49
status: hidden
category: model
summary: NVIDIA's family of open-weight reasoning models, combining Mamba, attention and mixture-of-experts layers for efficient agent workloads.
tags:
  - ModelRelease
  - OpenWeightLLM
  - ReasoningModels
---

**NVIDIA Nemotron 3** is a family of open-weight models for reasoning, coding and agent tasks, [announced on 15 December 2025](https://research.nvidia.com/labs/nemotron/Nemotron-3/). Nano launched first, with Super and Ultra following in 2026.

They combine Mamba, attention, and [Mixture of Experts Model](mixture-of-experts-model.md) layers, so a large total parameter count does not mean using every parameter for every token.

## Models

| Model and weights | Total parameters | Active parameters | Release date |
|---|---:|---:|---|
| [Nano 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 31.6B | 3.2B | 15 December 2025 |
| [Super 120B-A12B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 120B | 12B | 11 March 2026 |
| [Ultra 550B-A55B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) | 550B | 55B | 4 June 2026 |

Nano's 30B-A3B name is rounded. The [family research page](https://research.nvidia.com/labs/nemotron/Nemotron-3/) gives the more precise counts above, excluding embeddings from the active count; including them brings it to 3.6B. Super and Ultra's release dates come from their model cards.

These checkpoints take text input and produce text output, with support for up to 1M tokens of context. The deployed limit can be smaller: Nano's [default Hugging Face configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16#quick-start-guide) uses 256k because of memory requirements. Hosted endpoints have their own context and output limits.

## What is different?

The [hybrid architecture](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/) combines Mamba layers for efficient sequence processing, attention layers for accessing information across the context, and expert layers that activate only part of the model for each token.

Super and Ultra add **LatentMoE**, where experts work on a smaller internal representation, and **multi-token prediction**, which supports faster generation through speculative decoding. Both were pretrained using NVIDIA's NVFP4 format. The models also undergo [Reinforcement Learning (RL)](reinforcement-learning.md) across environments involving reasoning, coding and tool use, and support control over inference-time reasoning budgets. [NVIDIA's technical overview](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/) explains the design choices.

## Benchmarks

NVIDIA reports that Nano delivers 3.3 times the throughput of Qwen3-30B-A3B on a single H200, with 8k input and 16k output tokens. That is a result for a particular serving setup, not a universal speedup. [NVIDIA's launch results](https://research.nvidia.com/labs/nemotron/Nemotron-3/) include the comparison.

For example, its [Nano model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16#reasoning-benchmark-evaluations) reports 38.8% on [SWE-bench](swe-bench.md) using [OpenHands](openhands.md), compared with 22.0% for Qwen3-30B-A3B-Thinking-2507 and 34.0% for GPT-OSS-20B. Nano does not win everywhere: its MMLU-Pro score is below Qwen's.

The [Ultra model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16#benchmarks) reports 70.7% on [SWE-bench](swe-bench.md) Verified and 56.4% on [Terminal-Bench](terminal-bench.md) 2.1. Both trail Kimi-K2.6 in the same table. These are NVIDIA-reported results, and the harness, reasoning budget and evaluation setup matter when comparing them.

That is also why the family appears in [An Empirical Study of Harness Design for Coding Agents](an-empirical-study-of-harness-design-for-coding-agents.md). The study uses all three sizes to investigate how planning, tools and context management behave with different models. Its scores come from a different harness, so they should be read as a separate experiment.

## Availability

All three have downloadable weights. NVIDIA also publishes training recipes and datasets it can redistribute through its [Nemotron collection](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3).

The licences differ: Nano and Super use the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/), while Ultra's model card specifies [OpenMDW 1.1](https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1).

## Pricing and Cache Settings

Hosted pricing depends on the provider. These examples were checked on **23 September 2026**, in **USD per million tokens**:

| Model | Provider / route | Input | Output | Cached input |
|---|---|---:|---:|---:|
| Nano | [DeepInfra](https://deepinfra.com/nvidia/Nemotron-3-Nano-30B-A3B) | $0.05 | $0.20 | $0.025 |
| Super | [DeepInfra via OpenRouter](https://openrouter.ai/api/v1/models/nvidia/nemotron-3-super-120b-a12b/endpoints) | $0.085 | $0.40 | Not listed |
| Ultra | [DeepInfra](https://deepinfra.com/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B) | $0.50 | $2.20 | $0.10 |

These are standard rates for the listed routes. They serve a 262,144-token context, below the model's advertised 1M capacity. The listed Super route also caps output at 16,384 tokens.

DeepInfra's ordinary prompt caching is automatic and best-effort. Ultra additionally supports explicit retention for five minutes or one hour. Cache writes cost $0.625 or $1.00 per million tokens respectively, and operate in blocks of 8,192 tokens. Reusing the retained prefix costs the cached-input rate; extending the retention window incurs another write charge. See the [cache-retention documentation](https://docs.deepinfra.com/chat/prompt-cache-retention).

## Sources

- [NVIDIA Nemotron 3 family announcement](https://research.nvidia.com/labs/nemotron/Nemotron-3/)
- [Inside NVIDIA Nemotron 3: techniques, tools and data](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/)
- [Nemotron 3 Super announcement](https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/)
- [Nemotron 3 Ultra announcement](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/)
