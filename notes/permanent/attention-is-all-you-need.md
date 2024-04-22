---
title: Attention Is All You Need
date: 2023-12-04 00:00
modified: 2023-12-04 00:00
status: draft
---

Shell page for [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.

At the time of the paper, the prevalent sequence-to-sequence were [Encoder-Decoder](encoder-decoder.md) models that used recurrent or convolutional neural networks, where an encoder would represent an input sequence of tokens as a sequence of embeddings, and a decoder would take those embeddings and predict a new sequence one token at a time. The best performance models would connect the encoder and decoder using an [Attention](../../../permanent/attention.md) mechanism.

This paper proposes a revolutionary architecture called [Transformer](transformer.md), entirely based on attention mechanisms, without recurrence and convolutions at all.

The model achieved:

- 28.4 [Bilingual Evaluation Understudy](../../../permanent/bilingual-evaluation-understudy.md) on the [WMT 2014 English-to-German](WMT%202014%20English-to-German) translation task, improving over the existing best results, including ensembles. by over 2 [BLEU](BLEU).
 - On [WMT 2014 English-to-French](WMT%202014%20English-to-French) translation task, the model establishes a new single-model state-of-the-art BLEU score of 41.8 after 3.8 days of training on eight GPUs. That was a fraction of the training codes from literature.

They also show the Transformer generalised well to other tasks.