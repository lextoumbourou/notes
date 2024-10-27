---
title: Simple and Controllable Music Generation
date: 2023-12-02 00:00
modified: 2023-12-02 00:00
status: draft
---

*These are my notes from paper [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez.*

Note: this is unfinished. Some of the text is copied and pasted verbatim from the paper as I work through it. This is not a published draft.

## Abstract

* Introduces [MusicGen](../../../../permanent/musicgen.md) to tackle *conditional music generation*:
    * A [Language Model](../../permanent/language-model.md) that operates over RVQ tokens.
    * Comprised of a single-stage transformer LM together with efficient token interleaving patterns:
        * eliminates the need for cascading several streams
        * or cascading approaches like [Hierarchical Model](../../permanent/hierarchical-model.md) or Up-sampling.
    * Includes an algorithm for efficient [Token Interleaving Patterns](../../permanent/token-interleaving-patterns.md) so they don't need additional models for upsampling.
    * Can generate in mono and stereo.
    * Conditioned on text descriptions or melodic features.

## Introduction

* Generative music challenging due to difficulty modelling long range sequences.
* Music must have a higher [Sample Rate](../../permanent/sample-rate.md) than speech, as it requires full frequency spectrum.
    * 44.1kHz or 48 kHz vs 16 kHz for speech.
* Also complex structures: contains harmonies and melodies from different instruments:
    * Humans can't handle disharmony, so AI music has little room to make melodic errors.
* Utilises a number of recent advances:
    * self-supervised audio representation learning [A Cookbook of Self-Supervised Learning](paper-cookbook-of-self-supervised%20learning.md)
    * sequential modelling [LLaMA: Open and Efficient Foundation Language Models](../../../../permanent/llama-open-and-efficient-foundation-language-models.md)
    * audio synthesis [Survey on Neural Speech Synthesis](Survey%20on%20Neural%20Speech%20Synthesis)
    * [High Fidelity Neural Audio Compression](high-fidelity-neural-audio-compression.md) proposes [Encodec](../../permanent/encodec.md) which represents audio signals as multiple streams of discrete tokens.
         * Allows high-quality audio generation and effective audio modelling.
          * "However, this comes at the cost of jointly modelling several parallel dependent streams."
            * [MusicLM: Generating Music From Text](paper-musiclm-generating-music-from-text.md)
                * representing musical segments using multiple sequences of discrete tokens at different granularity
                * model them using a hierarchy of autoregressive models
            * In parallel, [Generating musical accompaniments from singing](Generating%20musical%20accompaniments%20from%20singing)
                * Similar approach for singing to accompaniment generation
            * Recently, [Neural Codec Language Models are Zero-Shot Text-to-Speech Synthesizers](neural-codec-language-models-are-zero-shot-text-to-speech-synthesizers.md)
                * proposed tackling this problem in two stages:
                    * (i) modelling the first stream of tokens only;
                    * (ii) then, apply a post-network to jointly model the rest of the streams in a non-autoregressive manner
* [MusicGen](../../../../permanent/musicgen.md)
    * a simple and controllable music generation model
    * generate high-quality music given textual description.
    * propose a general framework for modelling multiple parallel streams of acoustic tokens
        * generalisation of previous studies (see Fig 1).
        * ![](../../../../_media/papers-musicgen-fig-1.png)
    * can extend to stereo generate at no example cost.
    * also, introduces unsupervised melody conditioning, so model can generate music given harmonic and melodic structures.
    * Evaluation:
        * subjective rating of 84.8 out of 100 for MUSICGEN against 80.5 for the best baseline.
        * ablation study which sheds light on the importance of each of the components on the overall model performance.
        * Lastly, human evaluation suggests that MUSICGEN yields high quality samples which are better melodically aligned with a given harmonic structure, while adhering to a textual description.
    * Our contribution:
        * (i) We introduce a simple and efficient model to generate high quality music at 32 kHz.
            * We show that MUSICGEN can generate consistent music with a single-stage language model through an efficient codebook interleaving strategy.
        * (ii) We present a single model to perform both text and melody-conditioned generation and demonstrate that the generated audio is coherent with the provided melody and faithful to the text conditioning information.
        * (iii) We provide extensive objective and human evaluations on the key design choices behind our method.

## Method

* [MusicGen](../../../../permanent/musicgen.md)
    * autoregressive [Transformer](../../permanent/transformer.md)-based decoder, conditioned on a text or melody representation
    * A (language) model over the quantised units from [Encodec](../../permanent/encodec.md) audio tokeniser
        * high-fidelity reconstruction from a low frame rate discrete representation.
    * EnCodec returns several parallel streams.
        * Each stream is comprised of discrete tokens originating from different learned codebooks.
        * Prior work, proposed several modelling strategies to handle this issue.
        * In this work, we introduce a novel modelling framework, which generalises to various codebook interleaving patterns, and we explore several variants.
        * Through patterns, we can leverage the internal structure of the quantised audio tokens.
    * Supports conditional generation based on either text or melody.
* [Audio Tokenization](../../../../permanent/audio-tokenization.md)
    * Uses EnCodec, a convolutional auto-encoder with a latent space quantized using Residual Vector Quantization (RVQ)
    * And adversarial reconstruction loss.
    * [Encodec](../../permanent/encodec.md) overview
        * Given a reference audio random variable $X \in R \ d \cdot fs$
            * $d$ the audio duration
            * $fs$ the sample rate
        * Encode into a continuous tensor with a frame rate $fr << fs$
            * Representation is quantised into $Q \in \{1, ..., M\} \ d \ ·f r×K$
                * with K being the number of codebooks used in RVQ and M being the codebook size.
        * Notice, after quantisation we are left with K parallel discrete tokens sequences, each of length $T = d · fr$, representing the audio sample.
        * In $RVQ$, each quantiser encodes the quantisation error left by the previous quantiser
        * Thus quantised values for different codebooks are in general not independent, and the first codebook is the most important one.
* 2.2 [Codebook Interleaving Patterns](Codebook%20Interleaving%20Patterns) (see Figure 1)

Multiple techniqes

**Exact flattened autoregressive decomposition**

An autoregressive model requires a discrete random sequence $U \in \{1, ..., M\}^{s}$ with S the sequence length.

Take $U_0 = 0$, a special token indicating the beginning of a sequence. We can model the distribution as:

$\forall t > 0, p_t (U_{t-1,..., U_0}) \Delta \mathbb{P} [U_t | U_{t-1}, ..., U_o]$

Build a second sequence of random variables $\tilde{U}$ using the auto-regressive density p, e.g. we
define recursively $\tilde{U}_0 = 0$, and for all t > 0,

Now immediately have that $U$ and $\tilde{U}$ follow the same distriution. If we can fit a perfect estimate of $\hat{p}$ or $p$, then we can fit the distribiton of $U$.

As stated before, the main issue with the representation $Q$ we obtained from the EnCodec model is that there are K codebooks for each time step.

One solution would be to flatten out Q, thus taking $S = d \cdot f_r \cdot K$, e.g. first predicting the first codebook of the first time step, then the second codebook of the first time step, etc.

Then, using eq. (1) and eq. (2), we could theoretically fit an exact model of the distribution of Q. The downside however is the increased complexity, with part of the gain coming from the lowest sample rate fr being lost.

More than one possible flattening exists, and not all the pˆt functions need to be estimated through a single model.

For instance, MusicLM [Agostinelli et al., 2023] uses two models, one modeling the flattened first K/2 codebooks, and a second one the other K/2 flattened codebooks, conditioned on the decision of the first model. In that case, the number of autoregressive steps is still dfr · K.

**Inexact autoregressive decomposition**

Another possibility is to consider an autoregressive decomposition, where some codebooks are predicted in parallel. For instance, let us define another
sequence with V0 = 0 and for all t ∈ {1, . . . , T}, k ∈ {1, . . . , K}, Vt,k = Qt,k. When dropping the
codebook index k, e.g. Vt, we mean the concatenation of all the codebooks at time t.
pt,k (Vt−1, . . . , V0) ≜ P [Vt,k|Vt−1, ·, . . . , V0] . (3)
Let’s define again recursively V˜
0 = 0 and for all t > 0,
∀t > 0, ∀k, P
h
V˜
t,ki
= pt,k 
V˜
t−1, . . . , V˜
0

. (4)
Unlike in (2), we no longer have in the general case that V˜ follows the same distribution as V ,
even assuming we have access to the exact distribution pt,k. In fact, we would only have a proper
generative model if for all t, (Vt,k)k are independent conditionally on Vt−1, . . . , V0. As t increases,
the errors will compound and the two distributions can grow further apart. Such a decomposition
is inexact, but allows to keep the original frame rate which can considerably speed up training and
inference, especially for long sequences.

**Arbitrary codebook interleaving patterns**

In order to experiment with various such decompositions, and measure exactly the impact of using an inexact decomposition, we introduce codebook interleaving patterns.

Let us consider Ω = {(t, k) : {1, . . . , d · fr}, k ∈ {1, . . . , K}} be the set of all pairs of time steps and codebook indexes

A codebook pattern is a sequence P = (P0, P1, P2, . . . , PS), with P0 = ∅, and for all 0 < s ≤ S, Ps ⊂ Ω, such that P is partition of Ω. We model Q by predicting in parallel all the positions in Ps, conditionally on all the positions in P0, P1, . . . , Ps−1. Pragmatically, we restrict ourselves to patterns where each codebook index appears at most once in any of the Ps

We can now easily define a number of decompositions, for instance the “parallel” pattern given by
Ps = {(s, k) : k ∈ {1, . . . , K}}. (5)
It is also possible to introduce a “delay” between the codebooks, as in Kharitonov et al. [2022], e.g.,
Ps = {(s − k + 1, k) : k ∈ {1, . . . , K}, s − k ≥ 0}. (6)
Through empirical evaluations, we show the benefits and drawbacks of various codebook patterns,
shedding light on the importance of exact modeling of the parallel codebook sequences.

By convention, we will take U0 = 0, a deterministic special token indicating the beginning of the sequence. We can then model the distribution ∀t > 0, pt (Ut−1, . . . , U0) ≜ P [Ut|Ut−1, . . . , U0]
    * Arbitrary codebook interleaving patterns. In order to experiment with various such decompositions, and measure exactly the impact of using an inexact decomposition, we introduce codebook interleaving patterns. Let us consider Ω = {(t, k) : {1, . . . , d · fr}, k ∈ {1, . . . , K}} be the set of all pairs of time steps and codebook indexes. A codebook pattern is a sequence P = (P0, P1, P2, . . . , PS), with P0 = ∅, and for all 0 < s ≤ S, Ps ⊂ Ω, such that P is partition of Ω. We model Q by predicting in parallel all the positions in Ps, conditionally on all the positions in P0, P1, . . . , Ps−1. Pragmatically, we restrict ourselves to patterns where each codebook index appears at most once in any of the Ps. We can now easily define a number of decompositions, for instance the “parallel” pattern given by Ps = {(s, k) : k ∈ {1, . . . , K}}. (5) It is also possible to introduce a “delay” between the codebooks, as in Kharitonov et al. [2022], e.g., Ps = {(s − k + 1, k) : k ∈ {1, . . . , K}, s − k ≥ 0}. (6) Through empirical evaluations, we show the benefits and drawbacks of various codebook patterns, shedding light on the importance of exact modeling of the parallel codebook sequences
* 2.3 Model conditioning
    * Text conditioning. Given a textual description matching the input audio X, we compute a conditioning tensor C ∈ R TC ×D with D being the inner dimension used in the autoregressive model.
    * Generally, there are three main approaches for representing text for conditional audio generation. Kreuk et al. [2022] proposed using a pretrained text encoder, specifically T5 [Raffel et al., 2020].
    * Chung et al. [2022] show that using instruct-based language models provide superior performance.
    * Lastly, Agostinelli et al. [2023], Liu et al. [2023], Huang et al. [2023a], Sheffer and Adi [2023] claimed that joint text-audio representation, such as CLAP [Wu* et al., 2023], provides better-quality generations. We experiment with all of the above, respectively: T5 encoder, FLAN-T5, and CLAP.
    * Melody conditioning. While text is the prominent approach in conditional generative models nowadays, a more natural approach for music is conditioning on a melodic structure from another audio track or even whistling or humming. Such an approach also allows for an iterative refinement of the model’s output. To support that, we experiment with controlling the melodic structure via jointly conditioning on the input’s chromagram and text description. In preliminary experiments, we observed that conditioning on the raw chromagram often led to reconstructing the original sample, resulting in overfitting. To reduce it, we introduce an information bottleneck by choosing the dominant time-frequency bin in each time step. While a similar capability was shown in Agostinelli et al. [2023], the authors used supervised proprietary data, which is tedious and costly to collect. In this work, we take an unsupervised approach, eliminating the requirement for supervised data.
* 2.4 Model architecture
    * Codebook projection and positional embedding
        * Given a codebook pattern, only some codebooks are present at each pattern step Ps. We retrieve from Q the values corresponding to the indices in Ps. As noted in Section 2.2, each codebook is present at most once in Ps or not at all. If it is present, we use a learned embedding table with N entries and dimension D to represent the associated value from Q. Otherwise, we use a special token indicating its absence. We sum the contribution from each codebook after this transformation. As P0 = ∅, the first input is always the sum of all the special tokens. Finally, we sum a sinusoidal embedding to encode the current step s [Vaswani et al., 2017].
    * Transformer decoder.
        * The input is fed into a transformer with L layers and a dimension D. Each layer consists of a causal self-attention block. We then use a cross-attention block that is fed with the conditioning signal C. When using melody conditioning, we instead provide the conditioning tensor C as a prefix to the transformer input. The layer ends with a fully connected block consisting of a linear layer from D to 4·D channels, a ReLU, and a linear layer back to D channels. The attention and fully connected blocks are wrapped with a residual skip connection. Layer normalization [Ba et al., 2016] is applied to each block before being summed with the residual skip connection (“pre-norm”).
    * Logits prediction.
        * The output from the transformer decoder at pattern step Ps is transformed into logits prediction for the values of Q taken at the indices given by Ps+1. Each codebook is present at most once in Ps+1. If a codebook is present, the logits prediction is obtained by applying a codebook specific linear layer from D channels to N.
* 3 Experimental setup
    * 3.1 Models and hyperparameters
        * Audio tokenization model.
            * We use a non-causal five layers EnCodec model for 32 kHz monophonic audio with a stride of 640, resulting in a frame rate of 50 Hz, and an initial hidden size of 64, doubling at each of the model’s five layers.
            * The embeddings are quantized with a RVQ with four quantizers, each with a codebook size of 2048.
                * We follow Défossez et al. [2022] to train the model on one-second audio segments cropped at random in the audio sequence
        * Transformer model
            * We train autoregressive transformer models at different sizes:
                * 300M
                * 1.5B
                * 3.3B parameters.
            * Use Flash attention [Dao et al., 2022] a memory efficient from the xFormers package [Lefaudeux et al., 2022] to improve both speed and memory usage with long sequences.
            * We study the impact of the size of the model in Section 4.
