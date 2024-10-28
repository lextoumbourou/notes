---
title: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
date: 2024-10-25 00:00
modified: 2024-10-25 00:00
status: draft
---

## Overview

This paper proposes the [RNN Encoder-Decoder](../../permanent/rnn-encoder-decoder.md) neural network architecture for statistical machine translation. It uses two recurrent neural networks (RNNs) to encode a sequence of symbols into a fixed-length vector representation and then decode this representation back into another sequence of symbols. The encoder and decoder are jointly trained to maximise the probability of a target sequence given a source sequence. The paper demonstrates how this model can improve the performance of a statistical machine translation system by incorporating the conditional probabilities of phrase pairs computed by the RNN Encoder–Decoder as an additional feature in an existing log-linear model. The paper further shows that the RNN Encoder–Decoder learns a meaningful representation of linguistic phrases by capturing both semantic and syntactic structure.