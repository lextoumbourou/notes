---
title: "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
date: 2024-08-27 00:00
modified: 2026-09-26 08:55
status: draft
---

Introduces [Self-Instruct](self-instruct.md), a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations.

Pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model.
