---
title: Natural Language Processing with NLP - Week 2
date: 2022-10-07 00:00
status: draft
category: reference/moocs
parent: attention-models-in-nlp
---

# Week 2

## Week Introduction

* All about Transformer network.
    * Introduced in paper [Attention Is All Your Need](https://arxiv.org/abs/1706.03762).
* Despite the title, paper included convolutions.
    * These turned out not to be required.
* Many types of attention including: dot-product attention, casual attention, encoder-decoder attention and self-attention.
* This week will use the Transformer network for summarisation.

## Transformers vs RNNs

* With RNNs, sequential steps are required to encode input.
    * This means each word is dependant on the previous and it cannot be easily parallellised.
    * The longer the sequence, the slower the encoding. 
* You also run more risk of vanishing gradient problems.
    * LSTMs and GRUs help a bit, but even they struggle with very long sequences.
* Attention deals with vanishing gradients by using all inputs each step.
* Transformers only rely on attention, not recurrent networks.

## Transformers Overview

* Transformer architecture was described in the [Attention Is All Your Need](https://arxiv.org/abs/1706.03762) paper in 2017
* Standard for large language models include Bert, GPT3, T5 etc.
* Uses Scaled Dot-Product Attention:

![Scaled Dot-Product](/_media/attention-scaled-dot-product.png)

* Very efficient as it's just matrix multiplication operations and a Softmax.
* Model can grow larger and more complex while using less memory.

* Transformer model uses "multi-head attention" layer:
    * has multiple linear transformations and dot-product attention running in parallel.
    * linear transations have parameters that are learned.
    
    ![Multi-head attention](/_media/attention-multi-head.png)
    
* Encoder step-by-step:
    * Multi-head attention modules perform self-attention on input sequence.
        * Every item in input attends to every other item.
        * Then residual connection and normalisation.
        * Then feed forward layer
        * Then residual connection and normalization.
    * This layer is one block that is repeated N times.
    * Gives you contextual representation of inputs.
    
    ![Encoder diagram](/_media/attention-encoder.png)
* Decoder:
    ![Decoder](/_media/attention-decoder.png)
    * First attention module is masked so each position only attends to previous positions: can't see into the future.
    * 2nd attention module, takes encoder outputs and decoder attends to all items.
    * Whole later repeated one after the other.
* Positional encoding:
    * Transformers don't use recurrent neural networks, so needs a way to represent word order.
    * Positional encoding vectors are added to word embeddings for input embeddings.
    * Positional encoding can be learned or fixed.
    ![Positional Encoding](/_media/attention-pos-encoding.png)
