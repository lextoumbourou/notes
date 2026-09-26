---
title: "Soundstream: An End-to-End Neural Audio Codec"
date: 2024-01-12 00:00
modified: 2026-09-26 08:55
status: draft
---

These are my notes from the paper [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) by Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, Marco Tagliasacchi.

## Abstract

[SoundStream](soundstream.md) is a [Neural Audio Codec](neural-audio-codec.md) that uses a fully convolutional encoder/decoder network and a [Residual Vector Quantisation](residual-vector-quantization.md), which are trained jointly. They combine adversarial and reconstruction losses and can support generating high-quality audio from quantised embeddings.

They can operate across variable bitrates from 3kbps to 18kbps, with a negligible quality loss, compared with fixed bit rates, thanks to structured dropout.

They outperform the traditional codecs Opus and EVS.

## Introduction

Audio codecs can be partitioned into two broad categories: Waveform Codecs and Parametric Codecs.

Waveform codecs aim at producing at the decoder side a faithful reconstruction of the input audio samples.

In most cases, these codecs rely on transform coding techniques: a (usually invertible) transform is used to map an input time-domain waveform to the time-frequency domain. 

Then, transform coefficients are quantized and entropy coded.

At the decoder side the transform is inverted to reconstruct a time-domain waveform.

Often the bit allocation at the encoder is driven by a perceptual
model, which determines the quantization process. Generally,
waveform codecs make little or no assumptions about the
type of audio content and can thus operate on general audio.

As a consequence of this, they produce very high-quality
audio at medium-to-high bitrates, but they tend to introduce
coding artifacts when operating at low bitrates.

Parametric
codecs aim at overcoming this problem by making specific
assumptions about the source audio to be encoded (in most
cases, speech) and introducing strong priors in the form of a
parametric model that describes the audio synthesis process.

The encoder estimates the parameters of the model, which are
then quantized. The decoder generates a time-domain waveform
using a synthesis model driven by quantized parameters.

Unlike waveform codecs, the goal is not to obtain a faithful
reconstruction on a sample-by-sample basis, but rather to
generate audio that is perceptually similar to the original.

Traditional waveform and parametric codecs rely on signal
processing pipelines and carefully engineered design choices,
which exploit in-domain knowledge on psycho-acoustics and
speech synthesis to improve coding efficiency.

More recently, machine learning models have been successfully applied in the
field of audio compression, demonstrating the additional value
brought by data-driven solutions. 

For example, it is possible
to apply them as a post-processing step to improve the quality
of existing codecs.

This can be accomplished either via audio
superresolution, i.e., extending the frequency bandwidth [1],
via audio denoising, i.e., removing lossy coding artifacts [2],
or via packet loss concealment [3].

Other solutions adopt ML-based models as an integral part of
the audio codec architecture. 

In these areas, recent advances in
text-to-speech (TTS) technology proved to be a key ingredient.

For example, WaveNet [4], a strong generative model originally
applied to generate speech from text, was adopted as a decoder
in a neural codec [5], [6].

Other neural audio codecs adopt
different model architectures, e.g., WaveRNN in LPCNet [7]
and WaveGRU in Lyra [8], all targeting speech at low bitrates.

In this paper we propose SoundStream, a novel audio codec
that can compress speech, music and general audio more
efficiently than previous codecs, as illustrated in Figure 1.
SoundStream leverages state-of-the-art solutions in the field
of neural audio synthesis, and introduces a new learnable
quantization module, to deliver audio at high perceptual quality,
while operating at low-to-medium bitrates.

Figure 2 illustrates
the high level model architecture of the codec. 

A fully convolutional encoder receives as input a time-domain waveform and produces a sequence of embeddings at a lower sampling rate, which are quantized by [Residual Vector Quantisation](residual-vector-quantization.md).

Then a fully convolutional decoder receives the quantized embeddings and reconstructs an approximation of the original waveform.

The model is trained end-to-end using both reconstruction and adversarial losses.

To this end, one (or more) discriminators are trained jointly, with the goal of distinguishing the decoded audio from the original audio and, as a by-product, provide a space where a feature-based reconstruction loss can be computed.

The paper makes the following contributions:

- Both the encoder and the decoder only use causal convolutions, so the overall architectural latency of the model is determined solely by the temporal resampling ratio between the original time-domain waveform and the embeddings
- We propose SoundStream, a neural audio codec in which all the constituent components (encoder, decoder and quantizer) are trained end-to-end with a mix of reconstruction and adversarial losses to achieve superior audio quality, which enables a single model to handle different bitrates
- We demonstrate that learning the encoder brings a very significant coding efficiency improvement, with respect to a solution that adopts mel-spectrogram features
- We demonstrate by means of subjective quality metrics that SoundStream outperforms both Opus and EVS over a wide range of bitrates
- We design our model to support streamable inference, which can operate at low-latency. When deployed on a smartphone, it runs in real-time on a single CPU thread
- We propose a variant of the SoundStream codec that performs jointly audio compression and enhancement, without introducing additional latency.

## Related Work

### Traditional audio codecs

- [Opus Audio Codec](opus-audio-codec.md) and [Enhanced Voice Services](enhanced-voice-services.md) are state-of-the-art audio codecs, which combine traditional coding tools like LPC, CELP, and [Modified Discrete Cosine Transform](modified-discrete-cosine-transform.md) to deliver high coding efficiency over different content types, bitrates and sampling rates. We compare SoundStream with both Opus and EVS in subjective evaluation.
- Audio generative models
    - Several generative models have been developed for converting text or coded features into audio waveforms
        - WaveNet [4] allows for global and local signal conditioning to synthesize both speech and music
        - SampleRNN [11] uses recurrent networks in a similar fashion, but it relies on previous samples at different scales.
    - These auto-regressive models deliver very high-quality audio, at the cost of increased computational complexity, since samples are generated one by one.
    - To overcome this issue, Parallel WaveNet [12] allows for parallel computation, yielding considerable speedup during inference
    - Other approaches involve lightweight and sparse models [13] and networks mimicking the fast Fourier transform as part of the model [7]
    - More recently, generative adversarial models have emerged as a solution able to deliver high-quality audio with a lower computational complexity
    - MelGAN [15] is trained to produce audio waveforms when conditioned on mel-spectrograms, training a multi-scale waveform discriminator together with the generator.
    - HiFiGAN [16] takes a similar approach but it applies discriminators to both multiple scales and multiple periods of the audio samples
    - The design of the decoder and the losses in SoundStream is based on this class of audio generative models.
- Audio enhancement
    - Deep neural networks have been applied to different audio enhancement tasks, ranging from denoising [17]–[21] to dereverberation [22], [23], lossy coding denoising [2] and frequency bandwidth extension [1], 
    - In this paper we show that it is possible to jointly perform audio enhancement and compression with a single model, without introducing additional latency.
- Vector quantization
    - Learning the optimal quantizer is a key element to achieve high coding efficiency
    - Optimal scalar quantization based on Lloyd’s algorithm [25] can be extended to a high-dimensional space via the generalized Lloyd algorithm (GLA) [26], which is very similar to k-means clustering.
    - In vector quantization [28], a point in a high-dimensional space is mapped onto a discrete set of code vectors.
    - Vector quantization has been commonly used as a building block of traditional audio codecs [29].
    - For example, CELP [30] adopts an excitation signal encoded via a vector quantizer codebook
    - More recently, vector quantization has been applied in the context of neural network models to compress the latent representation of input features. 
    - For example, in variational autoencoders, vector quantization has been used to generate images [31], [32] and music [33], [34].
    -  Vector quantization can become prohibitively expensive, as the size of the codebook grows exponentially when rate is increased.
    - For this reason, structured vector quantizers [35], [36] (e.g., residual, product, lattice vector quantizers, etc.) have been proposed to obtain a trade-off between computational complexity and coding efficiency in traditional codecs
    - In SoundStream, we extend the learnable vector quantizer of [VQ-VAE](vq-vae.md) and introduce a residual (a.k.a. multi-stage) vector quantizer, which is learned end-to-end with the rest of the model.
    - To the best of the authors knowledge, this is the first time that this form of vector quantization is used in the context of neural networks and trained end-to-end with the rest of the model
- Neural audio codecs
    - End-to-end neural audio codecs rely on data-driven methods to learn efficient audio representations, instead of relying on handcrafted signal processing components.
    - Autoencoder networks with quantization of hidden features were applied to speech coding early on [37].
        - See [Speech coding based on a multi-layer neural network](https://ieeexplore.ieee.org/document/117117)
    - More recently, a more sophisticated deep convolutional network for speech compression was described in [38].
