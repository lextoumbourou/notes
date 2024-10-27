---
title: Attention Mechanism
date: 2022-10-07 00:00
status: draft
modified: 2023-04-16 00:00
---

**Attention Mechanism** or just **Attention** is a technique for calculating a weighted combination of a sequence of vectors, influence by human concept of attention.

It was originally devised as means of dealing with the a bottleneck in the [RNN Encoder-Decoder](rnn-encoder-decoder.md) architecture devised for Machine Translation, which at the time was looking like a promising alternative to the cumbersome statistical methods that were the current state of the art. In this seq-to-seq architecture, the encoder and decoder a are jointy trained to learn to represent input sequences, and decode that represenation into a translated seqeuence.

## Encoder-Decoder

Typically, the encoder would encode an input sequence into a fixed-length context vector, which is provided to the decoder as context for it to generate a sequence.

![](../../../_media/attention-mechanism-context-vector.png)

 However, a single context vector limited the encoders capaity to represent a sequence. As the sequences grew larger, the performance of these models would drop.

## Encoder-Decoder with Soft-Search

In Bahdanau et al, they proposed an alternative architecture: allow the encoder to represent each word in the input sequence as a vector. Then, in the decoder, it would calculate a context vector, which would be a weighted sum of the input sequence. The weights, or scores, would be calculated be a layer which would learn predict scores through training. They likened this to the idea of allowing the decoder to soft-search (soft because it can compute a score) through the input sequence at each decoder step.

![](../../../_media/attention-mechanism-attention.png)

Though the idea of attention had been explored in CV papers, this paper [Neural Machine Translation by Jointly Learning to Align and Translate (Sep 2014)](../reference/papers/neural-machine-translation-by-jointly-learning-to-align-and-translate-sep-2014.md) is typically credited as the origin of the attention mechanism [^1].

Allow the encoder to represent original sequence as a new encoded sequence, then add a mechanism to the decoder that allows it to "search through the encoded sequence" [^2], and assign weights to the most important words. They call this "soft search" and likened this to the human attribute attention, where we read by focusing on part of a text at a time.

Another side benefit of this, is that the calculated weights were interpretable, and allow for a visualisation of which input words most influenced each output word.

## Architecture

Shortly after this paper, many other varients of attention

This freed up the encoder from having to represent an entire sequence as a single vector, was very influential in neural machine translation, leading to the [Transformer](transformer.md) architecture years later.

Scaled-Dot Product Attention is a method of computing a token representation to include the context of surrounding tokens. It was described in the paper [Attention Is All You Need](attention-is-all-you-need.md) and is used in the [Transformer](../public/notes/permanent/transformer.md) architecture.
