---
title: Neural Audio Codec
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**Neural Audio Codec** is an audio codec that uses neural networks to compress and decompress audio, instead of the hand-designed signal processing used by traditional codecs like [Opus Audio Codec](opus-audio-codec.md) and [EVS](enhanced-voice-services.md).

A typical design has three parts: an encoder that turns the waveform into a compact sequence of embeddings, a quantiser (often [Residual Vector Quantisation](residual-vector-quantization.md)) that turns the embeddings into discrete codes, and a decoder that reconstructs the waveform from the codes. They're usually trained with a mix of reconstruction and adversarial losses.

Examples include [Soundstream: An End-to-End Neural Audio Codec](soundstream-an-end-to-end-neural-audio-codec.md), [Encodec](encodec.md) and [High-Fidelity Audio Compression with Improved RVQGAN](../reference/papers/high-fidelity-audio-compression-with-improved-rvqgan.md).

Since they turn audio into discrete tokens, they're also used as audio tokenisers for audio language models like [AudioLM: a Language Modeling Approach to Audio Generation](audiolm-a-language-modeling-approach-to-audio-generation.md) and [MusicGen](musicgen.md).
