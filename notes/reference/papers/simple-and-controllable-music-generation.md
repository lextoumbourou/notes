---
title: Simple and Controllable Music Generation
date: 2023-12-02 00:00
modified: 2023-12-02 00:00
status: draft
---

*These are my notes from paper [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez.*

Introduces [MusicGen](../../../../permanent/musicgen.md) to tackle *conditional music generation*: a [Language Model](../../permanent/language-model.md) that operates over [Residual Vector Quantisation](../../permanent/residual-vector-quantization.md) tokens.

Comprised of a single-stage transformer LM together with efficient token interleaving patterns:
* eliminates the need for cascading several streams
* or cascading approaches like [Hierarchical Model](../../permanent/hierarchical-model.md) or Up-sampling.
* Includes an algorithm for efficient [Token Interleaving Patterns](../../permanent/token-interleaving-patterns.md) so they don't need additional models for upsampling.
* Can generate in mono and stereo.
* Conditioned on text descriptions or melodic features.