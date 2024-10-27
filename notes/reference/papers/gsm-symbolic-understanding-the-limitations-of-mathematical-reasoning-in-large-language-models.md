---
title: "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models"
date: 2024-10-12 00:00
modified: 2024-10-12 00:00
status: draft
---

Notes from paper [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](https://arxiv.org/abs/2410.05229) by Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio and Mehrdad Farajtabar

## Overview

This research paper examines the limitations of mathematical reasoning in large language models (LLMs) when applied to school-level math problems. The authors introduce a benchmark called [GSM-Symbolic](../../permanent/gsm-symbolic.md), which generates variations of questions from the popular GSM8K dataset using symbolic templates. The study reveals significant performance variation across different instantiations of the same question, suggesting that LLMs may be relying on pattern-matching rather than genuine logical reasoning. The authors also find that LLM performance deteriorates as the complexity of the question increases, and that LLMs struggle to filter out irrelevant information, even when provided with multiple examples. This suggests that LLMs may not truly understand the underlying mathematical concepts and are prone to errors when presented with unfamiliar problem structures. The paper concludes that further research is necessary to develop LLMs capable of formal reasoning, moving beyond simplistic pattern recognition and towards more robust and generalisable problem-solving abilities.
