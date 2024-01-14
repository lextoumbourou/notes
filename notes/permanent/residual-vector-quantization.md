---
title: "Residual Vector Quantization"
date: 2024-01-13 00:00
modified: 2024-01-13 00:00
aliases:
  - Codebook
tags:
  - MachineLearning
  - AudioEngineering
cover: /_media/rvq-cover.png
summary: A technique for encoding audio into discrete tokens called *codes*
---

**Residual Vector Quantization** is a technique for encoding audio into discrete tokens called *codes*. That allows us to compress audio into small sizes - up to a 90x compression rate. But even more usefully, the discrete representations enable us to model audio using architectures that work on discrete representations, like Transformers; it's a tokeniser for audio. Now, we can make large language models for audio, speech or music. And that's exactly what AudioLM, MusicLM and MusicGen are.

RVQ was first described in the [Soundstream: An End-to-End Neural Audio Codec](../../../permanent/soundstream-an-end-to-end-neural-audio-codec.md) paper and has since been used in popular neural audio compression architectures like [SoundStream](../../../permanent/soundstream.md), [Encodec](../../../permanent/encodec.md) and [dac](../../../permanent/dac.md).

If we ignore the R part of RVQ, we have **Vector Quantization (VQ)**. [Quantization](../../../permanent/quantization.md) is the process of converting continuous values into discrete finite values, and **vector** quantisation is it applied to vectors. The idea of a VQ idea originally came from image modelling, by way of the [VQ-VAE](../../../permanent/vq-vae.md), and has recently been adapted to audio.

The VQ approach to encoding audio would look like this:

Before calling the VQ module, an encoder chunks an audio file into an array of vectors called a frame. The frame rate is model-dependent.

Now, we perform vector quantization for each frame by finding each closest neighbour in a lookup matrix called the **Codebook Table**. The position in the codebook table is a number called the code. Now, we can represent audio as a series of codes, a very compact representation of an audio file. We can swap this number for its vector representation for modelling.

We can train a model like this, by performing the encode and decode audio many times during training and calculate the reconstruction loss.

This diagram illustrates the training and inference architecture, as described by the SoundStream paper (fig. 2):

![](../_media/residual-vector-quantization-fig-2%201.png)

However, representing audio with a single code per frame will require an infinitely large codebook matrix. If we measure the difference between the encoded vector and the codebook vector, we get another vector that represents the difference or **Residual** between the two vectors. What if we used that vector to look up another codebook table? And, then, what if we repeated that process several times? We'd be sacrificing compression size for audio quality with each additional codebook.

So now we have <span style="color: red;">**Residual**</span> <span style="color: blue;">**Vector Quantization**</a>.

---

This article was heavily inspired by [What is Residual Vector Quanitzation](https://www.assemblyai.com/blog/what-is-residual-vector-quantization) by AssemblyAI.