---
title: Natural Language Processing with NLP - Week 1
date: 2022-10-04 00:00
status: draft
category: reference/moocs
parent: attention-models-in-nlp
---

## Intro

* Natural Language Processing with Attention Models.
* Instructors: Lukasz and Younes.
* Course includes:
    * state-of-the-art for practical NLP.
    * learn to build models from scratch.
    * also learn to fine state pretrained models (the "new normal" for modern deep learning).

## Week Introduction

* Week covers the problem of [[Machine Translation]] using attention.
* We know that an LSTM works for short to medium sequences, but for longer sequences faces problems of vanishing gradients.
* By adding an attention mechanism, the decoder can get access to all relevant parts of input sentence regardless of length.
* Many types of attention, but this week focuses on simple attention.
* Also covered:
    * greedy decoding.
    * random sampling.
    * beam search
        * minimise bias risk when predicting the next word in a translated sentence.

## Seq2Seq

* Outline:
    * Intro to Neural Machine Translation
    * Show model that has typically been used for it: the Seq2Seq model.
    * Look at model's deficiencies and the improvements made by subsequent models.

* Neural machine translation uses an encoder and decoder to translate languages.
    * English to German for this week's assignment.

* [Seq2Seq model by Google in 2014](https://arxiv.org/abs/1409.3215).
    * Takes a sequence of words (or any sequence you can encode as tokens) and return another sequence.
    * Works by mapping variable length sequences to fixed length memory called [[Embedding Space]].
    * Inputs and outputs don't need to be the same length.
    * LSTMs and GRUs can deal with vanishing and exploding gradients.
    * How it works
        * Encoder takes word tokens as inputs.
        * Returns a hidden state as output.
        * Hidden state is used by decoder to generate decoded sequence.
        ![Seq to seq encoder and decoder](_media/s2s-encoder-decoder.png)
    * Encoder
        * Has an embedding layer that converts input tokens to embedding vectors.
        * At each input step, LSTM gets input embedding and hidden state from previous step.
        * Encoder returns the final hidden state, which aims to encodes the meaning of the sequence.
        ![Seq to seq encoder](_media/s2s-encoder.png)

    * Decoder
        * Constructed of embedding layer and LSTM layer.
        * Use output of last hidden step of encoder and embedding for start token token `<sos>`.
        * Model outputs most probable next work, then pass the LSTM hidden state and embedding for predicted word to the next step.
            ![Seq2Seq Decoder](_media/s2s-decoder.png)

* Seq2Seq limitations:
    * Information bottleneck:
        * Since it uses a fixed length memory for hidden states, long sequences are difficult
        * As sequence size increases, model performance decreases.
        * Could work around using all encoder hidden states? Would have memory issues with bigger sequences.
        ![Information bottleneck](_media/s2s-info-bottle-neck.png)
* Attention:
    * Maybe model can learn what's important to focus on at each step?
    * More in next section.
    
        ![Seq 2 Seq attention idea](_media/s2s-attention.png)
    
