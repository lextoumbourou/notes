---
title: High Fidelity Neural Audio Compression
date: 2023-12-04 00:00
modified: 2023-12-04 00:00
status: draft
---

Shell page for [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) by Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi.

The paper that introduced [Encodec](../../permanent/encodec.md)

## Abstract

We introduce a state-of-the-art real-time, high-fidelity, audio codec leveraging neural networks.

It consists in a streaming encoder-decoder architecture with quantized latent space trained in an end-to-end fashion.

We simplify and speed-up the training by using a single multiscale spectrogram adversary that efficiently reduces artifacts and produce high-quality samples

We introduce a novel loss balancer mechanism to stabilize training: the weight of a loss now defines the fraction of the overall gradient it should represent, thus decoupling the choice of this hyper-parameter from the typical scale of the loss.

Finally, we study how lightweight Transformer models can be used to further compress the obtained representation by up to 40%, while staying faster than real time

We provide a detailed description of the key design choices of the proposed model including: training objective, architectural changes and a study of various perceptual loss functions.

We present an extensive subjective evaluation (MUSHRA tests) together with an ablation study for a range of bandwidths and audio domains, including speech, noisy-reverberant speech, and music.

Our approach is superior to the baselines
methods across all evaluated settings, considering both 24 kHz monophonic and 48 kHz
stereophonic audio.

Code and models are available at github.com/facebookresearch/encodec.

## 1 Introduction

Recent studies suggest that streaming audio and video have accounted for the majority of the internet traffic

in 2021 (82% according to (Cisco, 2021)).

With the internet traffic expected to grow, audio compression
is an increasingly important problem. In lossy signal compression we aim at minimising the bitrate of a sample while also minimising the amount of distortion according to a given metric, ideally correlated with human perception.

Audio codecs typically employ a carefully engineered pipeline combining an encoder and a decoder to remove redundancies in the audio content and yield a compact bitstream.

Traditionally, this is achieved by decomposing the input with a signal processing transform and trading off the quality of the components that are less likely to influence perception.

Leveraging neural networks as trained transforms via an encoder-decoder mechanism has been explored by Morishima et al. (1990); Rippel et al. (2019); Zeghidour et al. (2021). Our research work is in the continuity of this line of work, with a focus on audio signals.

The problems arising in lossy neural compression models are twofold: first, the model has to represent a
wide range of signals, such as not to overfit the training set or produce artifact laden audio outside its
comfort zone.

 We solve this by having a large and diverse training set (described in Section 4.1), as well
as discriminator networks (see Section 3.4) that serve as perceptual losses, which we study extensively in
Section 4.5.1, Table 2. T

The other problem is that of compressing efficiently, both in compute time and in size.
