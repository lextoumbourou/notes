---
title: VITS
date: 2023-12-23 00:00
modified: 2026-09-26 08:55
status: draft
---

**VITS** is an end-to-end text-to-speech model that generates raw audio waveforms directly from text, without a separate acoustic model and vocoder.

From the paper [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) by Jaehyeon Kim, Jungil Kong, and Juhee Son.

It combines a conditional [Variational Auto-Encoder](variational-auto-encoder.md) with normalising flows and adversarial training, similar to a [Generative Adversarial Network](generative-adversarial-network.md). It also uses a stochastic duration predictor, so the same text can be spoken with different rhythms.

VITS is the basis of voice conversion projects like [So-VITS-SVC](so-vits-svc.md).
