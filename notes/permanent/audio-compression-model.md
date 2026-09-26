---
title: Audio Compression Model
date: 2024-01-10 00:00
modified: 2026-09-26 08:55
status: draft
---

An **Audio Compression Model** is a model that compresses an audio waveform into a compact representation and can decode that representation back into audio.

Modern neural versions, like [SoundStream](soundstream.md) and [Encodec](encodec.md), use an encoder, a quantiser (typically [Residual Vector Quantisation](residual-vector-quantization.md)) and a decoder, trained end-to-end. See [Neural Audio Codec](neural-audio-codec.md).

As well as reducing bitrate, the discrete codes they produce can be used as tokens for audio language models, like in [AudioLM: a Language Modeling Approach to Audio Generation](audiolm-a-language-modeling-approach-to-audio-generation.md).
