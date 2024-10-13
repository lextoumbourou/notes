---
title: "Paper summary: Music Undersanding LLAMA: advancing text-to-music generation with question answering and captioning"
date: 2023-10-15 00:00
modified: 2023-10-15 00:00
status: draft
---

Links:

* Paper: https://arxiv.org/abs/2308.11276
* PDF: https://arxiv.org/pdf/2308.11276.pdf
* Code: https://github.com/crypto-code/MU-LLaMA
* Weights download: https://huggingface.co/mu-llama/MU-LLaMA/tree/main

Contributors

* Shansong Liu
* Atin Sakkeer Hussain
* Chenshuo Sun
* Ying Shan

Introduces [[Music Understanding LLaMA (MU-LLaMA)]].

A model capable of answering questions relating to musical audio and also captioning.

Utilises audio representations from a pretrained MERT model to extract music features (freezes the model).

Hard to find a suitable dataset for training the MU-LLaMA model.

So they generation question-answer pairs from existing audio captioning datasets.

Introduce: MusicQA dataset

Achieves good benchmarks in both music question answering and music caption generation.

## Introduction

For music generation: need of lots of music data with captions. Most of closed sourced due to license restrictions.

MusicCaps is largest. Only 28.52 hours of music accompanied by captions. Too small.

Urgent requirement: find a way to generate text-music pairs on a large scale for public use.

Song Describer1 is one of the few dedicated efforts to collect text-music pairs by crowd-sourcing for creating a public dataset.

The authors built an online platform to recruit volunteers to annotate their provided music. Nonetheless, this approach is time-consuming and uncontrollable, and not suitable for obtaining large quantities of data for T2M-Gen research.

As an alternative, we propose utilizing large language model (LLM) to automatically generate captions for vast amounts of music files from public resources.

Several models have been proposed for generating captions for music:
    - MusCaps [5]
    - audio captioning transformer [6]
    - audio captioning with aduio-text retriveal pretraining [7]
    - Whisper tiny audio captioning
    - LP-MusicCaps

Among these, MusCaps and LP-MusicCaps are currently the few specialized models dedicated explicitly to music captioning

The MusCaps model uses a hybrid architecture with a convolutional network for music understanding and recurrent neural network for captioning

The LP-MusicCaps model uses a cross-modal encoder-decoder architecture to understand and caption music.

An alternative approach for annotating music files involves employing audio question answering models to generate music captions

Recently, several multi-model and LLM based models capable of audio understanding and question answering have emerged such as LLaMA Adapter [9],UniVAL [10] and LTU [11].

LLaMA-adapter is a training scheme for finetuning LLMs. It offers multi-modal understanding capabilities based on ImageBind [12]. However, its ability to understand music is limited as it has not been trained on any music-text dataset.

UniVAL was designed as a universal model for image, video, audio and language tasks. However, UniVAL also lacks pretraining with music-text data.

The LTU model exhibits impressive performance and generalization abilities in
audio question answering. However, it should be noted that
the authors have not yet released the code and trained model
or their constructed OpenAQA-5M dataset.

Moreover, most
of the training data are regular audio files rather than music, so it is still not an appropriate method for music question answering and music captioning

In this paper, we present an innovative approach for
generating text-music pairs to advance the field of text-tomusic generation

To achieve this, we proposed a Music
Understanding LLM built upon the Meta’s LLaMA [13, 14]
model (MU-LLaMA) for music question answering

 The
proposed MU-LLaMA model3
is capable of generating captions through answering music-related questions for the given
music.

In order to train MU-LLaMA, we designed and constructed a MusicQA dataset from two publicly available
datesets, namely MusicCaps [4] and MagnaTagATune [15].

This paper contributes significantly to both the domains
of music question answering and text-to-music generation
in the following noteworthy ways:

1) We introduce the
MU-LLaMA model, an exceptional advancement capable of
performing music question answering and captioning tasks,
demonstrating superior performance across various metrics
over SOTA models;

2) We propose a systematic approach
for creating the music question answering dataset, crucial
for training the MU-LLaMA model;

 3) We demonstrate the
use of the MU-LLaMA model to generate music captions in
various formats required for developing T2M-Gen models.

This paper is organized as follows. In Section 2, we conduct a comprehensive comparison of different music representation methods to identify the most suitable music encoder
for our proposed MU-LLaMA model.

Section 3 outlines the
methodology for creating the MusicQA dataset, crucial for
training the MU-LLaMA model with the support of MosaicML’s MPT model [16].

Section 4 presents a detailed
exposition of the MU-LLaMA model’s structure and capabilities for music question answering and music captioning
tasks.

s. Experiments and evaluation of the MU-LLaMA model
are done in Section 5. Finally, the conclusion summarizes
key findings and contemplates potential future expansions.

## 2. MUSIC FEATURE REPRESENTATION

In order to equip our MU-LLaMA model with music understanding capabilities, we employ pretrained audio representation models.

These models can transform raw audio signals into meaningful representations that capture essential audio features, allowing machines to comprehend and interpret
sound information

In this section, we compare the following
audio representation models based on the performance of a
music tagging task on the MagnaTagATune [15] dataset.

1) ImageBind [12] is a method that learns a joint embedding space across 6 modalities, including images, text,
audio, depth, thermal and IMU data. It has the ability to represent and work with audio, enabling cross-modal retrieval
and composition of modalities through arithmetic operations.

2) JukeBox [17] utilizes a multi-scale VQ-VAE and autoregressive Transformers to represent and generate music. Its
music representation capabilities also extend to downstream tasks such as music information retrieval and music tagging.

3) OpenL3 [18] is a framework for learning deep audio embeddings through unsupervised audio-visual correspondence,
which can be used for various downstream tasks such as audio classification and analysis.

4) CLAP [19] leverages the publicly available LAION-Audio-630K dataset [19] for contrastive language-audio pretraining, facilitating cross-modal audio-text retrieval. 5

5) Wav2CLIP [20] is constructed by
distilling the CLIP [21] model, allowing audio to be projected into a shared embedding space together with images
and text

6) MERT [22], an acoustic music representation
model, achieves state-of-the-art performance on music understanding tasks through large-scale self-supervised training.

. It
utilizes teacher models, including an acoustic teacher based
on the RVQ-VAE model [23] and a musical teacher based on
the constant-Q transform (CQT) [24] feature.

From Table 1, MERT shows the best performance on the
downstream task of music tagging on the MagnaTagATune
(MTT) [15] dataset and hence we choose the MERT model to
generate music representation for our MU-LLaMA model.

3. MUSICQA DATASET GENERATION

In order to equip our proposed MU-LLaMA model with
music question answering capabilities, we require music
question-answer pairs for training the model.

 Existing publicly available music datasets typically consist of descriptions or tags, lacking ready-made music question-answer
pairs

Therefore, we propose an approach that leverages MosaicML’s MPT-7B model [16] to assist in generating music
question-answer pairs

The MPT model can generate desired
responses based on instructions.

Therefore, we devise a set
of instructions to generate music question-answer pairs from
music captioning and music tagging datasets.

The first set of instructions guides the MPT model to
generate answers based on the input caption or list of tags
for the following questions:
