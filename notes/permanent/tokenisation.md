---
title: Tokenisation
date: 2025-10-26 00:00
modified: 2026-09-26 08:55
status: draft
---

**Tokenisation** is the process of splitting input into a sequence of smaller units called tokens, which are then mapped to integer IDs that a model can process. It's usually the first step in [Text Preprocessing](text-preprocessing.md).

For text, tokens can be whole words (see [Word Tokenisation](word-tokenisation.md)), individual characters, or subword units. Modern language models mostly use subword tokenisers like Byte Pair Encoding, which keep common words as single tokens and split rare words into pieces.

The same idea applies to other modalities: see [Image Tokenisation](image-tokenisation.md) and [Audio Tokenisation](audio-tokenization.md).
