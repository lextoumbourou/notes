---
title: Acoustic Tokens
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**Acoustic Tokens** are discrete tokens produced by a neural audio codec, such as [SoundStream](soundstream.md) or [Encodec](encodec.md), that capture the fine acoustic details of an audio waveform: speaker identity, recording conditions and so on.

Because the codec is trained to reconstruct audio, acoustic tokens allow high-quality synthesis. However, a language model trained only on acoustic tokens struggles with long-term structure (for speech, it tends to produce babbling).

[AudioLM: a Language Modeling Approach to Audio Generation](audiolm-a-language-modeling-approach-to-audio-generation.md) combines them with semantic tokens, which capture long-term structure, to get both. See [Audio Tokenisation](audio-tokenization.md) and [Residual Vector Quantisation](residual-vector-quantization.md).
