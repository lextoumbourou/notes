---
title: "Neural Codec Language Models are Zero-Shot Text-to-Speech Synthesizers"
date: 2023-01-22 00:00
status: draft
category: reference/papers
tags:
  - MachineLearning
  - AudioEngineering
summary: "Notes from the paper [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111) by Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, Furu Wei"
---

Notes from the paper [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111) by Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, Furu Wei

---

## Overview

This paper describes a state-of-the-art (at least it was when it was released) [Zero-Shot TTS](../../permanent/zero-shot-tts.md) language model called [VALL-E](../../permanent/vall-e.md).

 At inference, VALL-E accepts text and a 3-second sample of a speaker and outputs new speech audio in that voice - even if the speaker was not in training data. The authors equate this capability with GPT-3's [In-Context Learning](../../permanent/in-context-learning.md) capability: learning information by providing additional context during inference.

## Key Details

### Use discrete codes as an intermediate representation of speech

The main breakthrough of this paper is utilising the **neural audio codec** [Encodec](../../../../permanent/encodec.md) (from the family of [RVQ](../public/notes/permanent/residual-vector-quantisation.md) models) to converts audio into discrete tokens, which they call *acoustic tokens*, allows them to model the problem as a language model—subsequently allowing them to train on larger, more noisy datasets than previous TTS solutions and seriously improving the ability to generate unseen speakers. They aim to see similar improvements in scaling up data NLP has enjoyed in the last few years.

Hence the paper the name: <font color="blue">Neural Codec</font> <font color="dark-yellow">Language Models</font> are <font color="orange">Zero-Shot</font> <font color="green">Text-to-Speech Synthesizes</font>.

They also use discrete codes to encode the speaker sample provided at inference time.
 
![](../../_media/neural-codec-language-models-are-zero-shot-text-to-speech-synthesizers-fig-1.png)

#### 2. Train on 60k hours of speech from the Libri-Light dataset

They train on 60k hours of audio from the [Libri-Light](https://github.com/facebookresearch/libri-light) dataset, hundreds of times more than existing TTS papers.

Since most of the data is unannotated, they use an off-the-shelf speech recognition model to generate the annotations ([Hybrid Deep Neural Network--Hidden Markov Model](Hybrid%20Deep%20Neural%20Network--Hidden%20Markov%20Model))).

#### 3. Model the codes with a Hierarchical Language Model Architecture

The tokens returned from RVQ have a hierarchical structure: tokens from the first quantisers can recover acoustic properties like speaker id, whereas later quantisers learn fine acoustic details. Therefore, they split the language model into two parts:

1. An [Autoregressive](Autoregressive) Transformer that is used to predict codes for the first codebook.

![](../../_media/neural-codec-language-models-are-zero-shot-text-to-speech-synthesizers-ar.png)
*Figure 3 from Neural Codec Language Models are Zero-Shot Text to Speech Synthesisers*


2. An [Non-Autoregressive](Non-Autoregressive) Transformer that predicts the subsequent codes from the first code. 

![](../../_media/neural-codec-language-models-are-zero-shot-text-to-speech-synthesizers-nar.png)
*Figure 3 from Neural Codec Language Models are Zero-Shot Text to Speech Synthesisers*

This configuration is a good trade-off between flexibility with the length of returned speech and inference performance, as the NAR can operate at O(1) instead of O(sequence length).

## History

VALL-E is a type of TTS model called **Cascading TTS System**, which means it uses an intermediate representation for audio, as opposed to end-to-end modelling.

---

In the past, a [Mel Spectrogram](../../permanent/mel-spectrogram.md) has been used as the intermediary representation, which relies on a Vocoder (like [HiFi-GAN](../../permanent/hifigan.md)) to the decoder. The problem is formulated as continuous signal regression.

However, these architectures typically need high-quality, clean audio to train on. And they don't tend to benefit from training on data scraped from the internet. Without the larger datasets, reliable zero-shot TTS on unseen speakers is impossible.

Language models are good at taking advantage of large datasets, especially unlabelled ones. To train a language model, a discrete representation is required, so the authors make use of recent quantisation models, namely [Encodec](../../../../permanent/encodec.md).

---

They utilise [LibriLight](https://github.com/facebookresearch/libri-light), which has 60K hours of English speech with over 7,000 unique speakers. They use a speech model to generate text transcriptions for all the audio. The dataset is noisier speech and inaccurate transcriptions than other TTS training datasets, like [LibriTTS](https://paperswithcode.com/dataset/libritts), but it has more diverse speakers and prosodies.

---

This capacity of in-context learning is mirrored with  [GPT-3](../../permanent/gpt-3.md).

Table 1 summarises the difference between VALL-E and previous TTS systems.

![](../../_media/neural-codec-language-models-are-zero-shot-text-to-speech-synthesizers-table-1%201.png)
*Table 1 from Neural Codec Language Models are Zero-Shot Text to Speech Synthesisers*

The acoustic tokens also allow us to generate diverse synthesised results in TTS by using different sampling strategies during inference.

---
## Results

They evaluate VALL-E on datasets where all test speakers are unseen in the training corpus, namely:
- [LibriSpeech](https://ieeexplore.ieee.org/document/7178964)
- [VCTK](https://datashare.ed.ac.uk/handle/10283/2651)

Results:
- Improve on SOTA TTS system [YourTTS](https://arxiv.org/abs/2112.02418) with improvements to speech naturalness
- LibriSpeech:
    - +0.12 [Comparitive Mean Opinion Score](Comparitive%20Mean%20Opinion%20Score) (CMOS), and speech naturalness
    - +0.93 [similarity-mean-option-score](../../../../permanent/similarity-mean-option-score.md) (SMOS).
- Beats the baseline on VCTK with +0.11 SMOS and +0.23 CMOS
- VCTK:
    - Gets +0.04 CMOS score against ground truth 
    - They claim the synthesised speech is as natural as the original recordings

Since VALL-E can synthesise diverse outputs with the same text and target speaker, it could be used for pseudo-data creation.

VALL-E was also found to keep the emotion of the acoustic prompt and the environment (reverberation, etc).

## Model

Both the AR model and the NAR model have the same [Transformer](../../permanent/transformer.md) architecture:

- 12 layers
- 16 attention heads
- an embedding dimension of 1024
- a feed-forward layer dimension of 4096
- dropout of 0.1.

Encodec is a convolutional RVQ model that can input and output 24 kHz audio across variable bitrates. It can produce embedding at 75 Hz for input waves at 24 kHz, a 320-fold reduction in sampling rate.

They use eight quantisers with 1024 code dimensionality.

![](../../_media/neural-codec-language-models-are-zero-shot-text-to-speech-synthesizers-fig-2.png)

For a 10-second audio waveform, the discrete representation would be $750 \times 8$ ($750 = \frac{24000 \times 10}{320}$), which for EnCodec, gives us a 6k bitrate for 24 kHz audio reconstruction. More quantities give better reconstruction quality.