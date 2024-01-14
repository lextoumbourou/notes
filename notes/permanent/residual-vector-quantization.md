---
title: Residual Vector Quantisation
date: 2024-01-13 00:00
modified: 2024-01-13 00:00
aliases:
  - Codebook
tags:
  - MachineLearning
  - AudioEngineering
cover: /_media/rvq-cover.png
summary: A tokeniser for audio
---

**Residual Vector Quantization (RVQ)** is a technique for encoding audio into discrete tokens called *codes*. It's a tokeniser for audio. Not only does that allow us to compress audio into absurdly small sizes - up to a 90x compression rate, but we can also use the tokens to model audio using the same architectures that work so well for text, like Transformers. Effectively, we can build large language models for audio, speech or music, and that's precisely what recent models like Google's [AudioLM](https://google-research.github.io/seanet/audiolm/examples/), Microsoft's [VALL-E (X)](https://www.microsoft.com/en-us/research/project/vall-e-x/) and Meta's [MusicGen](https://audiocraft.metademolab.com/musicgen.html) are.

RVQ applied to audio was first used to audio in the [Soundstream: An End-to-End Neural Audio Codec](../../../permanent/soundstream-an-end-to-end-neural-audio-codec.md) paper by Google Researchers and has since been used in popular neural audio compression architectures like [Encodec](../../../permanent/encodec.md) and [DAC](../../../permanent/dac.md).

To understand, RVQ. First, let's ignore the R part of RVQ. That leaves us with **Vector Quantization (VQ)**. Quantisation refers to converting infinite continuous values into discrete finite values, and here, as is the case in most ML, our inputs are [Vectors](vector.md). The VQ idea originally came from image tokenisation via the [VQ-VAE](../..///permanent/vq-vae.md) auto-encoder architecture.

## Vector Quantisation for Audio

The direct VQ approach to encoding audio would look like this:

1. **Audio Input**: an audio signal is represented as a multidimensional array of numbers with a known [Sample Rate](sample-rate.md) (usually 44100). A mono signal has one channel; stereo and others can have more.
2. **Encoder**: An encoder converts the signal into a sequence of vectors, one per "frame". The frame rate will be dependent on the model architecture and sample rate.
3. **Quantise**: Find the nearest neighbour in a lookup table called the codebook for each vector. The codebook table is learned alongside the encoder and decoder during training.
4. **Output**: The position of the lookup vector in the matrix is the "code" and is all we need to reconstruct the audio, given the RVQ model. However, we want to use the vector representation of codes for any upstream modelling.

![Vector Quantisation diagram](../_media/vector-quantisation.png)

However, representing audio with a single code per frame will never allow us to accurately reconstruct audio from these codes unless we have an infinitely large codebook matrix.

One clever technique is to take the difference between the encoded vector and the codebook vector, which we call the **Residual** vector. We can look that vector up in an additional codebook table. And we can repeat this process. Each time we do, we can reconstruct audio more accurately. However, it comes at the cost of poorer compression performance.

## Residual VQ

We add these steps to the VQ operation:

* **Residual** - calculate a difference vector called the Residual for each codebook vector and the input vector. Use that to look at a subsequent codebook.
* **Repeat** - repeat this for $Nq$ codebook tables.
* **Output** - at the end, we will have $Nq$ sequences of codes for modelling.

![Residual Vector Quanisation](../_media/residual-vector-quantisation.png)

So now we have: <span style="color: red;">**Residual**</span> <span style="color: blue;">**Vector Quantization**</a>.

## Training

We can train a model like this by performing the encode and decode audio during training and calculating various forms of reconstruction loss, including a GAN-style discriminator. This example is the architecture described in the [SoundStream](../../../permanent/soundstream.md) paper:

![SoundStream architecture](../_media/residual-vector-quantization-fig-2%201.png)

---

This article was heavily inspired by [What is Residual Vector Quanitzation](https://www.assemblyai.com/blog/what-is-residual-vector-quantization) by AssemblyAI.
