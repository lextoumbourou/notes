---
title: High-Fidelity Audio Compression with Improved RVQGAN
date: 2023-12-18 00:00
modified: 2023-12-18 00:00
status: draft
---

[High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546) with Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar

The paper that introduced [Descript Audio Codec (.dac)](https://github.com/descriptinc/descript-audio-codec)

## Main contributions

Introduce [[Improved RVQGAN]] a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality and fewer artifacts. Our model outperforms state-of-the-art methods by a large margin even at lower bitrates (higher compression) , when evaluated with both quantitative metrics and qualitative listening tests.

We identify a critical issue in existing models which don’t utilize the full bandwidth due
to codebook collapse (where a fraction of the codes are unused) and fix it using improved codebook learning techniques.

We identify a side-effect of quantizer dropout - a technique designed to allow a single
model to support variable bitrates, actually hurts the full-bandwidth audio quality and
propose a solution to mitigate it.

We make impactful design changes to existing neural audio codecs by adding periodic
inductive biases, multi-scale STFT discriminator, multi-scale mel loss and provide thorough ablations and intuitions to motivate them.

Our proposed method is a universal audio compression model, capable of handling speech, music, environmental sounds, different sampling rates and audio encoding formats.

## Introduction

Generative modeling of high-resolution audio is difficult due to high dimensionality (~44,100 samples
per second of audio) [24, 19], and presence of structure at different time-scales with both short and
long-term dependencies.

To mitigate this problem, audio generation is typically divided into two
stages: 1) predicting audio conditioned on some intermediate representation such as mel-spectrograms
[24, 28, 19, 30] and 2) predicting the intermediate representation given some conditioning information,
such as text [35, 34].

This can be interpreted as a hierarchical generative model, with observed
intermediate variables. Naturally, an alternate formulation is to learn the intermediate variables using
the variational auto-encoder (VAE) framework, with a learned conditional prior to predict the latent
variables given some conditioning. TThis formulation, with continuous latent variables and training an
expressive prior using normalizing flows has been quite successful for speech synthesis [17, 36].

A closely related idea is to train the same varitional-autoencoder with discrete latent variables using
VQ-VAE [38]. Arguably, discrete latent variables are a better choice since expressive priors can be
trained using powerful autoregressive models that have been developed for modeling distributions
over discrete variables [27].

Specifically, transformer language models [39] have already exhibited
the capacity to scale with data and model capacity to learn arbitrarily complex distributions such as
text[6], images[12, 44], audio [5, 41], music [1], etc

While modeling the prior is straightforward,
modeling the discrete latent codes using a quantized auto-encoder remains a challenge.

Learning these discrete codes can be interpreted as a lossy compression task, where the audio
signal is compressed into a discrete latent space by vector-quantizing the representations of an
autoencoder using a fixed length codebook

This audio compression model needs to satisfy the
following properties: 1) Reconstruct audio with high fidelity and free of artifacts 2) Achieve high
level of compression along with temporal downscaling to learn a compact representation that discards
low-level imperceptible details while preserving high-level structure [38, 33] 3)

Handle all types of
audio such as speech, music, environmental sounds, different audio encodings (such as mp3) as well
as different sampling rates using a single universal model.

While the recent neural audio compression algorithms such as SoundStream [46] and EnCodec [8]
partially satisfy these properties, they often suffer from the same issues that plague GAN-based
generation models. 

Specifically, such models exhibit audio artifacts such as tonal artifacts [29], pitch
and periodicity artifacts [25] and imperfectly model high-frequencies leading to audio that are clearly
distinguishable from originals

These models are often tailored to a specific type of audio signal such
as speech or music and struggle to model generic sounds. We make the following contributions:

We introduce Improved RVQGAN a high fidelity universal audio compression model, that
can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with
minimal loss in quality and fewer artifacts. Our model outperforms state-of-the-art methods
by a large margin even at lower bitrates (higher compression) , when evaluated with both
quantitative metrics and qualitative listening tests.
• We identify a critical issue in existing models which don’t utilize the full bandwidth due
to codebook collapse (where a fraction of the codes are unused) and fix it using improved
codebook learning techniques.
• We identify a side-effect of quantizer dropout - a technique designed to allow a single
model to support variable bitrates, actually hurts the full-bandwidth audio quality and
propose a solution to mitigate it.
• We make impactful design changes to existing neural audio codecs by adding periodic
inductive biases, multi-scale STFT discriminator, multi-scale mel loss and provide thorough
ablations and intuitions to motivate them.
• Our proposed method is a universal audio compression model, capable of handling speech,
music, environmental sounds, different sampling rates and audio encoding formats.

2 Related Work

High fidelity neural audio synthesis: Recently, generative adversarial networks (GANs) have
emerged as a solution to generate high-quality audio with fast inference speeds, due to the feedforward (parallel) generator

 MelGAN [19] successfully trains a GAN-based spectrogram inversion
(neural vocoding) model. It introduces a multi-scale waveform discriminator (MSD) to penalize
structure at different audio resolutions and a feature matching loss that minimizes L1 distance
between discriminator feature maps of real and synthetic audio.

HifiGAN [18] refines this recipe by
introducing a multi-period waveform discriminator (MPD) for high fidelity synthesis, and adding
an auxiliary mel-reconstruction loss for fast training. UnivNet [16] introduces a multi-resolution
spectrogram discriminator (MRSD) to generate audio with sharp spectrograms. BigVGAN [21]
extends the HifiGAN recipe by introducing a periodic inductive bias using the Snake activation
function [47]. It also replaces the MSD in HifiGAN with the MRSD to improve audio quality and
reduce pitch, periodicity artifacts [25]. While these the GAN-based learning techniques are used for
vocoding, these recipes are readily applicable to neural audio compression. Our Improved RVQGAN
model closely follows the BigVGAN training recipe, with a few key changes. Our model uses a
new multi-band multi-scale STFT discriminator that alleviates aliasing artifacts, and a multi-scale
mel-reconstruction loss that better model quick transients.

Neural audio compression models: VQ-VAEs [38] have been the dominant paradigm to train neural
audio codecs. The first VQ-VAE based speech codec was proposed in [13] operating at 1.6 kbps. This
model used the original architecture from [38] with a convolutional encoder and an autoregressive
1
https://github.com/descriptinc/descript-audio-codec
2
https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5
2
wavenet [27] decoder

SoundStream [46] is one of the first universal compression models capable
of handling diverse audio types, while supporting varying bitrates using a single model. They use a
fully causal convolutional encoder and decoder network, and perform residual vector quantization
(RVQ). 

The model is trained using the VQ-GAN [12] formulation, by adding adversarial and featurematching losses along with the multi-scale spectral reconstruction loss.

EnCodec [8] closely follows
the SoundStream recipe, with a few modifications that lead to improved quality. EnCodec uses a multiscale STFT discriminator with a multi-scale spectral reconstruction loss

 They use a loss balancer
which adjusts loss weights based on the varying scale of gradients coming from the discriminator.





## How 

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --args.load conf/ablations/baseline.yml --save_path runs/baseline/
```