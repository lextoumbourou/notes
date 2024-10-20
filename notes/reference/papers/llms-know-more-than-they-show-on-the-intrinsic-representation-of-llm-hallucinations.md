---
title: "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations"
date: 2024-10-07 00:00
modified: 2024-10-07 00:00
status: draft
tags:
- MachineLearning
- LargeLanguageModels
---

These are my notes from the paper [LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations](https://arxiv.org/abs/2410.02707) by Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, Yonatan Belinkov.

These notes are incomplete.

## Overview

[Large Language Models](../../permanent/large-language-models.md)s produces errors, like "factual inaccuracies, biases, and reasoning failures". We refer to these errors as [Hallucinations](../../../../permanent/Hallucinations.md).

Recent work has shown that LLMs encode information in their internal states about the truthfulness of their outputs. This information can be used to detect errors.

This paper shows that these internal representations encode more information about truthfulness than we previously thought. They find truthfulness information is concentrated in specific tokens, which can be used to enhance error detection performance.

However, the error detectors fail to generalise across datasets, implying that [Truthfulness Encoding](../../permanent/truthfulness-encoding.md) is not universal but rather multifaceted. They also show a discrepancy between LLMs' internal encoding and external behaviour: they may encode the correct answer yet consistently generate an incorrect one.

These insights deepen our understanding of LLM errors from the model's internal perspective, which can guide future research on enhancing error analysis and mitigation.
