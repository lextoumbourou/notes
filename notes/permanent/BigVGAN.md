---
title: BigVGAN
date: 2023-12-24 00:00
modified: 2026-09-26 08:55
status: draft
---

**BigVGAN** is a universal neural vocoder: a [Generative Adversarial Network](generative-adversarial-network.md)-based model that turns [Mel Spectrogram](mel-spectrogram.md) into raw audio waveforms, and is trained at large scale so it generalises to speakers, languages and recording conditions it didn't see in training.

It builds on [HiFi-GAN](hifigan.md), and replaces the usual activations with the [Snake Activation Function](snake-activation-function.md), which adds a [Periodic Inductive Biases](Periodic%20Inductive%20Biases.md) that suits audio.

Idea from paper *BigVGAN: A Universal Neural Vocoder with Large-Scale Training*.
