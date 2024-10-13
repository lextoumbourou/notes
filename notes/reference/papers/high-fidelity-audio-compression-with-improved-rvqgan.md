---
title: High-Fidelity Audio Compression with Improved RVQGAN
date: 2023-12-18 00:00
modified: 2023-12-18 00:00
status: draft
---

[High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546) with Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar.

---

## Overview

This paper introduces an audio compression architecture that can compress 44.1Khz audio into an 8kbps bitrate (~90x compression).

The author's make the weights and code available on GitHub [.dac](https://github.com/descriptinc/descript-audio-codec)

## Main Contributions

The authors use a encoder/decoder convolution architecture with [Residual Vector Quantisation](../../permanent/residual-vector-quantization.md), which was originally used in [SoundStream](../../../../permanent/soundstream.md) and later [Encodec](../../permanent/encodec.md). The architecture is called Improved RVQGAN, although it's commonly referred to as DAC, based on the repository name.

Improved RVQGAN makes these architectural and training improvements:

* Replace [Leaky ReLU](../../permanent/leaky-relu.md) with the [Snake Activation Function](../../permanent/snake-activation-function.md) which is helps to with the periodic nature of audio.
* Two changes to Vector Quantisation operation based on ideas from [Improved VQGAN](../../../../permanent/improved-vqgan.md):
    * Project the query embedding into low-dimensional space before performing the nearest neighbour lookup for codes (64d to 8d), this decouples code lookup and code embedding. We can think of it as using the principle components to do the lookup.

      ```python
      input_dim = 64
      codebook_dim = 8
      in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
      
      # in forward pass:
      z_e = self.in_proj(z)
      ```

    * L2-normalisation of encoded and codebook vectors, converting lookup from euclidean distance to cosine similarity.
* The original RVQ proposal includes codebook dropout, so the model is sometimes reconstructing audio using only some of the codebooks. They found this hurts the model performance when using all codebooks, so they only do this 50% of the time.
* For the discriminator they use a Multi-Scale Time-Frequency Spectrogram Discriminator
* They continue to use multiple loss functions, but include multi-scale mel loss.

They use multiple loss functions:
* [Frequency Domain Reconstruction Loss](Frequency%20Domain%20Reconstruction%20Loss)
* [Adversarial Loss](Adversarial%20Loss)
* [Codebook Learning](Codebook%20Learning)
* Weighting it 15, 2, 1, 1, 0.25, respectively.

---

## Introduction

Generative modelling of audio is difficult for 2 key reasons:
* audio is extremely high dimensionality data (~44,100 samples per second of audio)
* audio has complicated structure that includes short and long term dependancies. For example, the pluck of a guitar happens in a fraction of a second, whereas the arrangement of an entire composition spans the duration of the song.

A common solution is to divide audio generation in 2 stages:
1. predict audio conditioned on an intermediate representation.
2. predict intermediate representation given some conditioning information like text.

If the intermediate representation is some kind of discrete "codes" ie tokens, we can use them for upstream modelling using architectures like transformers that work well in many other problems.

The process of learrning discrete codes is really just to compression.

The audio signal is compressed into a discrete latent space using [Residual Vector Quantisation](../../permanent/residual-vector-quantization.md) vector-quantizing the representations of an autoencoder using a fixed length codebook.

* Generative modelling of high-resolution audio is difficult because:
    * high dimensionality (~44,100 samples per second of audio)
        * See [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](../../../../permanent/samplernn-an-unconditional-end-to-end-neural-audio-generation-model.md)
        * [Generative adversarial networks for conditional waveform synthesis](Generative%20adversarial%20networks%20for%20conditional%20waveform%20synthesis)
    * Structure at different time-scales with short and long term dependencies.
* Common mitigations:
    * audio generation is typically divided into two stages:
        * 1. predicting audio conditioned on some intermediate representation such as mel-spectrograms see:
            * [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](../../../../permanent/samplernn-an-unconditional-end-to-end-neural-audio-generation-model.md)
            * [Deep voice 3: Scaling text-to-speech with convolutional sequence learning](Deep%20voice%203:%20Scaling%20text-to-speech%20with%20convolutional%20sequence%20learning)
            * [Generative adversarial networks for conditional waveform synthesis](Generative%20adversarial%20networks%20for%20conditional%20waveform%20synthesis)
            * [Waveglow: A flow-based generative network for speech synthesis](Waveglow:%20A%20flow-based%20generative%20network%20for%20speech%20synthesis)
        * 2. predicting the intermediate representation given some conditioning information, such as text
            * [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](../../../../permanent/natural-tts-synthesis-by-conditioning-wavenet-on-mel-spectrogram-predictions.md)
            * [Fastspeech 2: Fast and high-quality end-to-end text to speech](Fastspeech%202:%20Fast%20and%20high-quality%20end-to-end%20text%20to%20speech)
    * Can be interpret this as a [Hierarchical Generative Model](../../../../permanent/hierarchical-generative-model.md) with observed intermediate variables.
* Alternate formulation is to learn the intermediate variables using a [Variational Auto-Encoder](../../../../permanent/variational-auto-encoder.md) framework with a learned conditional prior to predict the latent variables given some conditioning.
    * This formulation, with continuous latent variables and training an expressive prior using normalizing flows has been quite successful for speech synthesis.
        * [Conditional variational autoencoder with adversar- ial learning for end-to-end text-to-speech](Conditional%20variational%20autoencoder%20with%20adversar-%20ial%20learning%20for%20end-to-end%20text-to-speech)
        * [Naturalspeech: End-to-end text to speech synthesis with human-level quality.](Naturalspeech:%20End-to-end%20text%20to%20speech%20synthesis%20with%20human-level%20quality.)
* Closely related idea:
    * train the same varitional-autoencoder with discrete latent variables using VQ-VAE: [Neural discrete representation learning. Advances in neural information processing systems](Neural%20discrete%20representation%20learning.%20Advances%20in%20neural%20information%20processing%20systems)
* [Discrete Latent Variables](Discrete%20Latent%20Variables) could be a better choice since expressive priors can be trained using powerful autoregressive models that have been developed for modelling distributions over discrete variables.
    * See: [Wavenet: A generative model for raw audio](Wavenet:%20A%20generative%20model%20for%20raw%20audio)
    * We know that [Transformer](../../permanent/transformer.md) language models can scale with data and learn complex distributions like text, images, audio, music, etc
* Modelling the prior is straightforward but modelling the discrete latent codes using a quantized auto-encoder remains a challenge.
* Learning discrete codes can be interpreted as a lossy compression task:
    * audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencoder using a fixed length codebook.
* An [Audio Compression Model](../../../../permanent/audio-compression-model.md) needs to satisfy the following properties:
    * Reconstruct audio with high fidelity and free of artifacts
    * Achieve high level of compression along with temporal downscaling to learn a compact representation that discards low-level imperceptible details while preserving high-level structure:
        * [Advances in neural information processing systems](Advances%20in%20neural%20information%20processing%20systems)
        * [Generating Diverse High-Fidelity Images with VQ-VAE-2](../../../../permanent/generating-diverse-high-fidelity-images-with-vq-vae-2.md)
    * Handle all types of audio such as speech, music, environmental sounds, different audio encodings (such as mp3) as well as different sampling rates using a single universal model.
* Some audio compression models like [SoundStream](../../../../permanent/soundstream.md) and [Encodec](../../permanent/encodec.md) partially satisfy these properties but suffer from issues that plauge GAN models:
    * have audio artifacts such as tonal artifacts
        * See [Upsampling artifacts in neural audio synthesis](Upsampling%20artifacts%20in%20neural%20audio%20synthesis)
    * Pitch and periodicty artifacts
        * See [Chunked Autoregressive GAN for Conditional Waveform Synthesis](../../../../permanent/chunked-autoregressive-gan-for-conditional-waveform-synthesis.md)
    * Not doing a good job of modelling high-frequencies
    * These models are often tailored to a specific type of audio signal, like speech or music and struggle to model generic sounds
* They make the following contributions:
    * Introduce [Improved RVQGAN](Improved%20RVQGAN) a high fidelity universal audio compression model, that:
        * can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality and fewer artifacts.
            * outperforms state-of-the-art methods by a large margin even at lower bitrates (higher compression) when evaluated with both quantitative metrics and qualitative listening tests.
    * Identify a critical issue in existing models which don’t utilize the full bandwidth due to codebook collapse (where a fraction of the codes are unused) and fix it using improved codebook learning techniques.
    * Identify side-effect of quantizer dropout - a technique designed to allow a single model to support variable bitrates, actually hurts the full-bandwidth audio quality and propose a solution to mitigate it.
    * We make impactful design changes to existing neural audio codecs by adding:
        * [Periodic Inductive Biases](../../../../permanent/Periodic%20Inductive%20Biases.md)
        * [multi-scale-stft-discriminator](../../../../permanent/multi-scale-stft-discriminator.md)
        * [Multi-scale Mel Loss](Multi-scale%20Mel%20Loss)
    * Provide thorough ablations and intuitions to motivate them.
* Proposed method: universal audio compression model, capable of handling speech, music, environmental sounds, different sampling rates and audio encoding formats.

## Related Work

### High fidelity neural audio synthesis

[Generative Adversarial Network](../../../../permanent/generative-adversarial-network.md) models are a solution to generate high-quality audio with fast inference speeds, due to the feedforward (parallel) generator:

* [MelGAN](../../../../permanent/MelGAN.md)
        - Successfully trains a GAN-based spectrogram inversion (neural vocoding) model
        - Introduces:
            - [Multi-scale Waveform Discriminator](Multi-scale%20Waveform%20Discriminator) (MSD)
                - penalize structure at different audio resolutions
            - feature matching loss that minimises L1 distance between discriminator feature maps of real and synthetic audio.
* [HiFi-GAN](../../permanent/hifigan.md)
    * Introduce a multi-period waveform discriminator (MPD) for high fidelity synthesis
    * adding an auxiliary mel-reconstruction loss for fast training
* [univnet](../../../../permanent/univnet.md)
    * introduces a multi-resolution spectrogram discriminator (MRSD) to generate audio with sharp spectrograms
* [BigVGAN](../../../../permanent/BigVGAN.md)
    * Improve HifiGAN recipe by introducing a periodic inductive bias using the [Snake Activation Function](../../permanent/snake-activation-function.md)
        * [Neural networks fail to learn periodic functions and how to fix it](../../../../permanent/neural-networks-fail-to-learn-periodic-functions-and-how-to-fix-it.md)
    * Replaces the MSD in HifiGAN with the MRSD to improve audio quality and reduce pitch, periodicity artifacts
        * See [Chunked Autoregressive GAN for Conditional Waveform Synthesis](../../../../permanent/chunked-autoregressive-gan-for-conditional-waveform-synthesis.md)

The GAN-based learning techniques have been used for vocoding, but they also work for [Neural Audio Compression](Neural%20Audio%20Compression).

[Improved RVQGAN](Improved%20RVQGAN) model closely follows the BigVGAN training recipe, with a few key changes:
* Uses a new multi-band, multi-scale [STFT Discriminator](../../../../permanent/STFT%20Discriminator.md) that alleviates aliasing artifacts
* A multi-scale mel-reconstruction loss that better models quick transients.

Neural audio compression models: VQ-VAEs have been the dominant paradigm to train neural audio codecs.

First VQ-VAE based speech codec was proposed in [Low bit-rate speech coding with vq-vae and a wavenet decoder](Low%20bit-rate%20speech%20coding%20with%20vq-vae%20and%20a%20wavenet%20decoder) operating at 1.6 kbps

This model used the original architecture from [Neural discrete representation learning](Neural%20discrete%20representation%20learning) with a convolutional encoder and an autoregressive [Wavenet](../../permanent/wavenet.md) decoder.

[SoundStream](../../../../permanent/soundstream.md)
* one of the first audio compression models that can handle diverse audio types with varying bitrates on a single model.
* Use a fully causal convolutional encoder and decoder network, and perform [Residual Vector Quantisation](../../permanent/residual-vector-quantization.md)
* The model is trained using the VQ-GAN formulation by adding adversarial and feature matching losses along with the multi-scale spectral reconstruction loss
    * See [Taming Transformers for High-Resolution Image Synthesis](../../../../permanent/taming-transformers-for-high-resolution-image-synthesis.md)

[Encodec](../../permanent/encodec.md)
* Follows the SoundStream recipe, with a few modifications that lead to improved quality:
    * uses a multi-scale [STFT Discriminator](STFT%20Discriminator) with a multi-scale spectral reconstruction loss
    * Also use a loss balancer to adjust loss weights based on the varying scale of gradients coming from the discriminator.

Propose method shares these ideas:
* Use convolutional encoder-decoder architecture
* [Residual Vector Quantisation](../../permanent/residual-vector-quantization.md)
* Adversarial, perceptual losses.

Has these differences:
* Introduce periodic inductive bias using [Snake Activations](Snake%20Activations)
* Improve codebook learning by projecting encodings into a low-dimensional space.
* Obtain a stable training recipe using best practices for adversarial and perceptual loss design, with fixed loss weights and without needing a sophisticated loss balancer.

These changes allow for near optimal effective bandwidth usage.

Allows model to outperform Encodec with 3x lower bitrate.

---

Language modelling of natural signals

Neural language models have demonstrated great success in diverse tasks such as open-ended text generation with in-context learning capabilities.
* See [Language models are few shot learners](Language%20models%20are%20few%20shot%20learners).

[Self-Attention](../../permanent/self-attention.md) allows it to complex, long-range dependencies. However, has quadratic computational cost with the length of the sequence. Bad for natural signals like images, audio with high dimensionality. Instead they need a compact mapping into a discrete representation space.

Mapping usually learned with [VQ-GAN](../../../../permanent/vq-gan.md), followed by training autoregressive Transformer on discrete tokens.
* [Taming Transformers for High-Resolution Image Synthesis](../../../../permanent/taming-transformers-for-high-resolution-image-synthesis.md).
* [Vector-quantized image modeling with improved vqgan](Vector-quantized%20image%20modeling%20with%20improved%20vqgan)

This approach has shown success across image, audio, video and music domains

Codecs like SoundStream and EnCodec have already been used in generative audio models:
* AudioLM [a language modeling approach to audio generation](a%20language%20modeling%20approach%20to%20audio%20generation)
* MusicLM [Musiclm: Generating music from text](Musiclm:%20Generating%20music%20from%20text)
* VALL-E [Neural codec language models are zero-shot text to speech synthesizers](Neural%20codec%20language%20models%20are%20zero-shot%20text%20to%20speech%20synthesizers)

DAC is a drop-in replacement for the audio tokenization model used in these methods with
* highly superior audio fidelity
* efficient learning due to our maximum entropy code representation

## The Improved RVQGAN Model

Like [SoundStream](../../../../permanent/soundstream.md) and [Encodec](../../permanent/encodec.md), uses an RVQGAN architecture which is built on framework of [VQ-GAN](../../../../permanent/vq-gan.md) models

* Architecture: Full Convolutional [Encoder-Decoder](../../permanent/encoder-decoder.md) like [SoundStream](../../../../permanent/soundstream.md)
* Goal: time-based downscaling with a chosen striding factor
* Special techniques:
    * Quantise the encoding with [Residual Vector Quantisation](../../permanent/residual-vector-quantization.md)
        * "recursively quantises residuals following an initial quantisation step with a distinct codebook"
* Apply quantizer dropout, so that some of the later codebooks are not always used, which comes from SoundStream.
* Loss: [Frequency Domain Reconstruction Loss](Frequency%20Domain%20Reconstruction%20Loss) and adversarial and perceptual losses.
* Inputs:
     * Audio signal with sampling rate $fs$ (Hz)
     * Encoding striding factor $M$
     * $Nq$ layers of $RVQ$
* Output:
    * Discrete code matrix of shape $S \ \times \ Nq$
        * $S$ is the frame rate defined as $fs/M$

Note: target bitrate is upper bound, since all models support variable bitrates.

Table 1 shows [Improved RVQGAN](Improved%20RVQGAN) again baseline comparing compression factors and frame rate of latent codes.

![](../../../../_media/high-fidelity-audio-compression-with-improved-rvqgan-table1.png)

Model achieves:
* higher compression factor
* outperforms in audio quality (shown later)

Lower frame rate is desirable when training a language model on the discrete codes, as it results in shorter sequences.

### Periodic activation function (Snake Activation Function)

The authors note that audio waveforms have high "periodicity" (in other words, the waveform repeats itself a bunch). The most common neural network activation for generative models is [Leaky ReLU](../../permanent/leaky-relu.md) but it struggles to extrapolate periodic signals, causes poor generalisation.

By replacing Leaky ReLU with the [Snake Activation Function](../../permanent/snake-activation-function.md), it adds "periodic inductive bias" to the generator. The BigVGAN model introduced the Snake Activation function to the audio domain.

In the ablation studies, snake activation function was an important change in improving audio fidelity.

Defined as $\text{snake}(x) = x + \frac{1}{\alpha} \sin^2(\alpha)$
* $\alpha$ controls the frequency of periodic component of the signal
* In experiments, replacing Leaky ReLU activations with Snake function is influential change that significantly improves audio fidelity (Table 2).

![](../../../../_media/high-fidelity-audio-compression-with-improved-rvqgan-table2.png)

#### Improved residual vector quantization

Vector quantisation (VQ) is commonly used to train discrete auto-encoders, but they have challenges.

* Vanilla VQ-VAEs struggle from low codebook usage
    * due to poor initialization, leading to a significant portion of the codebook being unused.
* This reduction in effective codebook size leads to an implicit reduction in target bitrate, which translates to poor reconstruction quality
* To mitigate this, recent audio codec methods use kmeans clustering to initialize the codebook vectors, and manually employ randomized restarts [9] when certain codebooks are unused for several batches.
    * See [Jukebox: A generative model for music](Jukebox:%20A%20generative%20model%20for%20music)

However, they find that the EnCodec model trained at 24kbps target bitrate, as well as our proposed model with the same codebook learning method (Proposed w/ EMA) still suffers from codebook under-utilization (Figure 1).

To address this issue, use two key techniques to improve codebook usage:
* factorized codes
* L2-normalized codes.

 Factorization decouples code lookup and code embedding, by performing code lookup in a low-dimensional space (8d or 32d) whereas the code embedding resides in a high dimensional space (1024d).

Intuitively, this can be interpreted as a code lookup using only the principal components of the input vector that maximally explain the variance in the data.

The L2-normalization of the encoded and codebook vectors converts euclidean distance to cosine similarity, which is helpful for stability and quality [44].
See [Vector-quantized image modeling with improved vqgan](Vector-quantized%20image%20modeling%20with%20improved%20vqgan)]

These two tricks along with the overall model recipe significantly improve codebook usage, and therefore bitrate efficiency (Figure 1) and reconstruction quality (Table 2), while being simpler to implement.

Our model can be trained using the original VQ-VAE codebook and commitment losses. See [Neural discrete representation learning](Neural%20discrete%20representation%20learning)

The equations for the modified codebook learning procedure are written in Appendix A.

### 3.3 Quantiser dropout rate

Quantiser dropout was introduced in SoundStream, which allows a single model to support variable bitrates, as the model learns to reconstruct audio using only some of the codebooks.

The number of quantisers $Nq$ determine the bitrate, so for each input example we randomly sample $n ∼ {1, 2, . . . , nq}$ and only use the first $nq$ quantizers while training.

However, the authors found that this causes the audio reconstruction to degrade when you have full bandwidth. See Fig 2 below.

![](../../../../_media/high-fidelity-audio-compression-with-improved-rvqgan-fig2-1.png)
However, we noticed that applying quantizer dropout degrades the audio reconstruction quality at full bandwidth (Figure 2)

What they do is only apply the dropout operation 50% of the time.

Now they have the best of both worlds: at lower bitrates, the audio can be constructed well, but at maximum bitrates they get close to optimal reconstruction.

They found this techinque causes the quantized codes to learn most-significant to least significant bits of information with each additional quantizer. When the codes are reconstructed with $1 . . . Nq$ codebooks, we can see each codebook adds increasing amounts of fine-scale detail.
This interaction is useful to understand when training hierarchical generative models, like [AudioLM](../../permanent/audiolm.md), [VALL-E](../../permanent/vall-e.md) and [MusicLM](MusicLM). Could consider partitioning the codes into "coarse" tokens (most significant codes) and "fine" tokens (higher detail, but less significant).

#### 3.4 Discriminator design

Like prior work, we use multi-scale (MSD) and multi-period waveform discriminators (MPD) which lead to improved audio fidelity

However, spectrograms of generated audio can still appear blurry, exhibiting over-smoothing artifacts in high frequencies
* See [A neural vocoder with multi-resolution spectrogram discriminators for high-fidelity waveform generation](A%20neural%20vocoder%20with%20multi-resolution%20spectrogram%20discriminators%20for%20high-fidelity%20waveform%20generation)

The multi-resolution spectrogram discriminator (MRSD) was proposed in UnivNet to fix these artifacts and BigVGAN [21] found that it also helps to reduce pitch and periodicity artifacts

However, using magnitude spectrograms discards phase information which could’ve been otherwise utilized by the discriminator to penalize phase modeling errors.

Moreover, we find that high-frequency modeling is still challenging for these models especially at high sampling rates

To address these issues, we use a complex STFT discriminator [46] at multiple time-scales [8] and find that it works better in practice and leads to improved phase modeling.

Additionally we find that splitting the STFT into sub-bands slightly improves high frequency prediction and mitigates aliasing artifacts, since the discriminator can learn discriminative features about a specific sub-band and provide a stronger gradient signal to the generator.

Multi-band processing was earlier proposed in [43] to predict audio in sub-bands which are subsequently summed to produce the full-band audio.

#### 3.5 Loss functions

Frequency domain reconstruction loss: while the mel-reconstruction loss [18] is known to improve stability, fidelity and convergence speed, the multi-scale spectral losses[42, 11, 15] encourage modeling of frequencies in multiple time-scales.

In our model, we combine both methods by using a L1 loss on mel-spectrograms computed with window lengths of [32, 64, 128, 256, 512, 1024, 2048] and hop length set to window_length / 4.

We especially find that using the lowest hop size of 8 improves modeling of very quick transients that are especially common in the music domain.

EnCodec [8] uses a similar loss formulation, but with both L1 and L2 loss terms, and a fixed mel bin size of 64.

We find that fixing mel bin size leads to holes in the spectrogram especially at low filter lengths.

Therefore, we use mel bin sizes [5, 10, 20, 40, 80, 160, 320] corresponding to the above filter lengths which were verified to be correct by manual inspection.

Adversarial loss: our model uses the multi-period discriminator [18] for waveform discrimination, as well as the proposed multi-band multi-scale STFT discriminator for the frequency domain.

We use the HingeGAN [22] adversarial loss formulation, and apply the L1 feature matching loss [19].

Codebook learning: we use the simple codebook and commitment losses with stop-gradients from the original VQ-VAE formulation [38], and backpropagate gradients through the codebook lookup using the straight-through estimator [3].

Loss weighting: we use the loss weightings of 15.0 for the multi-scale mel loss, 2.0 for the feature matching loss, 1.0 for the adversarial loss and 1.0, 0.25 for the codebook and commitment losses respectively

These loss weightings are in line with recent works [18, 21] (which use 45.0 weighting for the mel loss), but simply rescaled to account for the multiple scales and log10 base we used for computing the mel loss. We don’t use a loss balancer as proposed in EnCodec [8].

### Experiments

### Data sources

Train on dataset of speech, music, and environmental sounds.

For speech:
* [DAPS dataset](https://ieeexplore.ieee.org/document/6981922)
* Clean speech segments from [DNS Challenge 4](Icassp 2022 deep noise suppression challenge)
* [Common Voice](Common voice: A massively-multilingual speech corpus) dataset
* [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset

For music:
* [MUSDB](https://sigsep.github.io/datasets/musdb.html) dataset (150 tracks, around 10h of music)
* [MTG-Jamendo Dataset](https://mtg.github.io/mtg-jamendo-dataset/) (55k tracks)

Environment sounds:
* Use both the balanced and unbalanced train segments from [AudioSet](https://ieeexplore.ieee.org/document/7952261)

Preprocessing:
    - Resample to 44kHz.

Training process:
* Extract short excepts from each audio file
* normalize to -24 dB LUFS

Data augmentation:
* Randomly shift phase of excerpt, uniformly.

Evaluation:
* Evaluation segments from [AudioSet](https://ieeexplore.ieee.org/document/7952261)
* Two speakers that are held out from DAPS (F10, M10) for speech
* Test split of MUSDB
* Extract 3000 10-second segments (1000 from each domain) as test set.

### Balanced data sampling

We take special care in how we sample from our dataset.

Though our dataset is resampled to 44kHz, the data within it may be band-limited in some way

That is, some audio may have had an original sampling rate much lower than 44kHz

This is especially prevalent in speech data, where the true sampling rates of the underlying data can vary greatly (e.g. the Common Voice data is commonly (8-16kHz)

When we trained models on varying sampling rates, we found that the resultant model often would not reconstruct data above a certain frequency.

When investigating, we found that this threshold frequency corresponded to the average true sampling rate of our dataset. To fix this, we introduce a balanced data sampling technique.

We first split our dataset into data sources that we know to be full-band - they are confirmed to contain energy in frequencies up to the desired [Nyquist-Shannon Sampling Theorem](../../permanent/nyquist-shannon-sampling-theorem.md) frequency (22.05kHz) of the codec - and data sources where we have no assurances of the max frequency

When sampling batches, we make sure that a full-band item is sampled.

Finally, we ensure that in each batch, there are an equal number of items from each domain: speech, music, and environmental sound.

In our ablation study, we examine how this balanced sampling technique affects model performance.

### 4.3 Model and training recipe

Model architecture:
* convolutional encoder
* a residual vector quantizer
* convolutional decoder

The basic building block of our network is a convolutional layer which either upsamples or downsamples with some stride, followed by a residual layer consisting of convolutional layers interleaved with non-linear Snake activations

Our encoder has 4 of these layers, each of which downsamples the input audio waveform at rates `[2, 4, 8, 8]`.

Our decoder has 4 corresponding layers, which upsample at rates `[8, 8, 4, 2]`.

We set the decoder dimension to 1536.

Total parameters: 76M (22M in encoder and 54M in decoder)

They also try decoder dimensions of 512 (31M parameters) and 1024 (49M params).

Uses:
* multi-period discriminator (see [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646).
    * use periods of `[2, 3, 5, 7, 11]`
* complex multi-scale STFT discriminator.
    * use window lengths of `[2048, 1024, 512]`
    * hop-length: 1/4 window length.

For band-splitting of the STFT, we use the bandlimits [0.0, 0.1, 0.25, 0.5, 0.75, 1.0].

For the reconstruction loss:
* use distance between log-mel spectrograms with window lengths [32, 64, 128, 256, 512, 1024, 2048], with corresponding number of mels for each of [5, 10, 20, 40, 80, 160, 320].
* The hop length is 1/4 of the window length
* We use feature matching and codebook losses, as described in Section 3.5.
* For our ablation study, we train each model with a batch size of 12 for 250k iterations.
* In practice, this takes about 30 hours to train on a single GPU. For our final model, we train with a batch size of 72 for 400k iterations
* We train with excerpts of duration 0.38s. We use the AdamW optimizer [23] with a learning rate of 1e − 4, β1 = 0.8, and β2 = 0.9, for both the generator and the discriminator.
* We decay the learning rate at every step, with γ = 0.999996.

### 4.4 Objective and subjective metrics

Use the following metrics:
* ViSQOL [7]: an intrusive perceptual quality metric that uses spectral similarity to the ground truth to estimate a mean opinion score
* Mel distance: distance between log mel spectrograms of the reconstructed and ground truth waveforms.
    * The configuration of this loss is the same as described in 3.5.
* STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms.
    * We use window lengths [2048, 512]
    * This metric captures the fidelity in higher frequencies better than the mel distance.
* Scale-invariant source-to-distortion ratio (SI-SDR) (from [Sdr–half-baked or well done?](https://arxiv.org/abs/1811.02508))
    * distance between waveforms, similar to signal-to-noise ratio, with modifications so that it is invariant to scale differences.
    * When considered alongside spectral metrics, SI-SDR indicates the quality of the phase reconstruction of the audio.
* Bitrate efficiency: We calculate bitrate efficiency as the sum of the entropy (in bits) of each codebook when applied on a large test set divided by the number of bits across all codebooks.
* For efficient bitrate utilization this should tend to 100% and lower percentages indicate that the bitrate is being underutilized.
* Conduct a MUSHRA-inspired listening test, with a hidden reference, but no low-passed anchor
    * In it each one of ten expert listeners rated 12 randomly selected 10-second samples from our evaluation set, 4 of each domain; speech, music and environmental sounds

Make comparisons at:
    - Improved RVQGAN: 2.67kbps, 5.33kbps and 8kbps
    - EnCodec: 3kbps, 6kbps and 12kbps.

### 4.5 Ablation study

The results of our ablation study can be seen in Table 2.

Use 4 objectives to compare modules.

Architecture: We find that varying the decoder dimension has some effect on performance, with smaller models having consistently worse metrics

However, the model with decoder dimension 1024 has similar performance to the baseline, indicating that smaller models can still be competitive.

Biggest impact: relu activation for the snake activation

This change resulted in much better SI-SDR and other metrics.

Similar to the results in [BigVGAN](Bigvgan: A universal neural vocoder with large-scale training), find periodic inductive bias for snake activation helpful for waveform generation.

For our final model, we use the largest decoder dimension (1536), and the snake activation.

Discriminator: Next, we removed or changed the discriminators one-by-one, to see their impact on the final result. First, we find that the multi-band STFT discriminator does not result in significantly better metrics, except for SI-SDR, where it is slightly better. However, when inspecting spectrograms of generated waveforms, we find that the multi-band discriminator alleviates aliasing of high frequencies.

* The upsampling layers of the decoder introduce significant aliasing artifacts [29]
* The multi-band discriminator is more easily able to detect these aliasing artifacts and give feedback to the generator to remove them. Since aliasing artifacts are very small in terms of magnitude, their effect on our objective metrics is minimal. Thus, we keep the multi-band discriminator.

We find that adversarial losses are critical to both the quality of the output audio, as well as the bitrate efficiency. When training with only reconstruction loss, the bitrate efficiency drops from 99% to 62%, and the SI-SDR drops from 9.12 to 1.07. The other metrics capture spectral distance, and are relatively unaffected. However, the audio from this model has many artifacts, including buzzing, as it has not learned to reconstruct phase. Finally, we found that swapping the multi-period discriminator for the single-scale waveform discriminator proposed in MelGAN [19] resulted in worse SI-SDR. We retain the multi-period discriminator.

Impact of low-hop reconstruction loss: We find that low-hop reconstruction is critical to both the waveform loss and the modeling of fast transients and high frequencies. When replaced with a single-scale high-hop mel reconstruction (80 mels, with a window length of 512), we find significantly lower SI-SDR (7.68 from 9.12). Subjectively, we find that this model does much better at capturing certain sorts of sounds, such as cymbal crashes, beeping and alarms, and singing vocals. We retain the multi-scale mel reconstruction loss in our final recipe.

Latent dimension of codebook: the latent dimension of the codebook has a significant impact on bitrate efficiency, and consequently the reconstruction quality. If set too low or too high (e.g. 2, 256), quantitative metrics are significantly worse with drastically lowered bitrate efficiency. Lower bitrate efficiency results in effectively lowered bandwidth, which harms the modeling capability of the generator. As the generator is weakened, the discriminator tends to “win”, and thus the generator does
not learn to generate audio with high audio quality. We find 8 to be optimal for the latent dimension.

Quantization setup: we find that using exponential moving average as the codebook learning method, as in EnCodec[8], results in worse metrics especially for SI-SDR. It also results in poorer codebook utilization across all codebooks (Figure 1)

When taken with its increased implementation complexity (requiring K-Means initialization and random restarts), we retain our simpler projected lookup method for learning codebooks, along with a commitment loss.

Next, we note that the quantization dropout rate has a significant effect on the quantitative metrics. However, as seen in Figure 2, a dropout of 0.0 results in poor reconstruction with fewer codebooks

As this makes usage of the codec challenging for downstream generative modeling tasks, we instead use a dropout rate of 0.5 in our final model. This achieves a good trade-off between audio quality at full bitrate as well as lower bitrates. Finally, we show that we can increase the max bitrate of our model from 8kbps to 24kbps and achieve excellent audio quality, surpassing all other model configurations. However, for our final model, we train at the lower bitrates, in order to push the compression rate as much as possible.

Balanced data sampling: When removed, this results in worse metrics across the board. Empirically, we find that without balanced data sampling, the model produces waveforms that have a max frequency of around 18kHz.

This corresponds to the max frequency preserved by various audio compression algorithms like MPEG, which make up the vast majority of our datasets. With balanced data sampling, we sample full-band audio from high-quality datasets (e.g. DAPS) just as much as possibly band-limited audio from datasets of unknown quality (e.g. Common Voice). This alleviates the issue, allowing our codec to reconstruct full-band audio, as well as band-limited audio.

#### 4.6 Comparison to other methods

We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra
[46], and Opus [37], a popular open-source audio codec

For EnCodec, Lyra, and Opus, we use
publicly available open-source implementations provided by the authors.

We compare using both
objective and subjective evaluations, at varying bitrates

The results are shown in Table 3. We find
that the proposed codec out-performs all competing codecs at all bitrates in terms of both objective
and subjective metrics, while modeling a much wider bandwidth of 22kHz.

In Figure 3, we show the result of our MUSHRA study, which compares EnCodec to our proposed
codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than
EnCodec at all bitrates. However, even at the highest bitrate, it still falls short of the reference
MUSHRA score, indicating that there is room for improvement. We note that the metrics of our final
model are still lower than the 24kbps model trained in our ablation study, as can be seen in Table 2.
This indicates that the remaining performance gap may be closed by increasing the maximum bitrate

In Figure 4 and Table 4, we compare our proposed model trained with the same exact configuration as
EnCodec (24 KHz sampling rate, 24 kbps bitrate, 320 stride, 32 codebooks of 10 bits each) to existing
baselines, in both quantitative and qualitative metrics. In Figure 5, we show qualitative results by
sound category.

## 5 Conclusion

We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data.
Our method combines the latest advancements in audio generation, vector quantization techniques,
and improved adversarial and reconstruction losses. Our extensive evaluation against existing audio compression algorithms demonstrates the superiority of our approach, providing a promising
foundation for future high-fidelity audio modeling. With thorough ablations, open-source code, and
trained model weights, we aim to contribute a useful centerpiece to the generative audio modeling
community

Broader impact and limitations: our model has the capability to make generative modeling of
full-band audio much easier to do. While this unlocks many useful applications, such as media editing,
text-to-speech synthesis, music synthesis, and more, it can also lead to harmful applications like
deepfakes. Care should be taken to avoid these applications. One possibility is to add watermarking
and/or train a classifier that can detect whether or not the codec is applied, in order to enable the
detection of synthetic media generated based on our codec. Also, our model is not perfect, and still
has difficulty reconstructing some challenging audio. By slicing the results by domain we find that,
even though the proposed codec outperforms competing approaches across all of the domains, it
performs best for speech and has more issues with environmental sounds. Finally, we notice that it
does not model some musical instruments perfectly, such as glockenspeil, or synthesizer sounds.
