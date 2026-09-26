---
title: Machine Translation
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**Machine Translation** is the task of automatically translating text or speech from one language to another.

Early systems were rule-based, followed by statistical approaches that learned translation probabilities from large parallel corpora. Neural machine translation took over in the mid-2010s, first with [RNN Encoder-Decoder](rnn-encoder-decoder.md) models, then with the [Attention Mechanism](attention-mechanism.md) introduced in [Neural Machine Translation by Jointly Learning to Align and Translate (Sep 2014)](../reference/papers/neural-machine-translation-by-jointly-learning-to-align-and-translate-sep-2014.md), and then with the [Transformer](transformer.md) from [Attention Is All You Need](attention-is-all-you-need.md).

Translation quality is commonly measured with the [BLEU Score](bleu-score.md). [Backtranslation](backtranslation.md) is a common trick for generating extra training data.
