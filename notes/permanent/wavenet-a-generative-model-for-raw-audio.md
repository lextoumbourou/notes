---
title: "WaveNet: A Generative Model for Raw Audio"
date: 2024-01-07 00:00
modified: 2026-09-26 08:55
status: draft
---

**WaveNet: A Generative Model for Raw Audio** is a 2016 paper from DeepMind by Aäron van den Oord and others, which introduced [Wavenet](wavenet.md).

Notes from paper [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

WaveNet generates raw audio waveforms one sample at a time, with each sample conditioned on all the previous samples (autoregressive). To cover a long enough history of samples, it uses stacks of dilated causal convolutions, whose receptive field grows exponentially with depth.

It produced much more natural-sounding text-to-speech than previous systems, and can also generate music. It was later used as the vocoder in [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](natural-tts-synthesis-by-conditioning-wavenet-on-mel-spectrogram-predictions.md).
