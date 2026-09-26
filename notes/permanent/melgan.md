---
title: MelGAN
date: 2023-12-24 00:00
modified: 2026-09-26 08:55
status: draft
---

**MelGAN** is a non-autoregressive, fully convolutional [Generative Adversarial Network](generative-adversarial-network.md)-based vocoder that converts [Mel Spectrogram](mel-spectrogram.md) into raw audio waveforms.

The generator is trained against a multi-scale waveform discriminator, which judges the audio at several different resolutions. Because it isn't autoregressive, it's much faster than models like [WaveNet: A Generative Model for Raw Audio](wavenet-a-generative-model-for-raw-audio.md), and later vocoders like [HiFi-GAN](hifigan.md) build on the same approach.

Model from paper *MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis*.
