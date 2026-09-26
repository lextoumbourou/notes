---
title: "Audio Tokenisation"
date: 2024-03-09 00:00
modified: 2026-09-26 08:55
status: draft
---

**Audio Tokenisation** is the process of converting audio into a sequence of discrete tokens, so that language modelling techniques can be applied to audio. See [Tokenisation](tokenisation.md).

[AudioLM: a Language Modeling Approach to Audio Generation](audiolm-a-language-modeling-approach-to-audio-generation.md) describes two common kinds of audio tokens:

* [Acoustic Tokens](acoustic-tokens.md), produced by a neural audio codec like [SoundStream](soundstream.md) or [Encodec](encodec.md). They capture fine acoustic detail and allow high-quality reconstruction.
* Semantic tokens, produced by clustering the embeddings of a self-supervised model like [w2v-BERT](w2v-bert.md) or [HuBERT](hubert.md) with [K-Means](k-means.md). They capture long-term structure like phonetic content but reconstruct poorly.
