---
title: Denoising Diffusion Probabilistic Models
date: 2024-03-14 00:00
modified: 2026-09-26 08:50
status: draft
tags:
- DiffusionModels
---

Notes from paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by Jonathan Ho, Ajay Jain, Pieter Abbeel.

## Overview

This research paper presents a significant advancement in the field of diffusion probabilistic models, a type of generative model inspired by non-equilibrium thermodynamics.

The authors demonstrate the potential of these models for generating high-quality images, surpassing even the results achieved by generative adversarial networks (GANs) in some cases. They introduce a novel connection between diffusion models and denoising score matching, revealing a simplified and weighted variational bound objective. This equivalence facilitates the training of diffusion models, leading to enhanced sample quality. The authors also explore the concept of progressive lossy compression, highlighting the inherent inductive bias of diffusion models that makes them effective lossy compressors. By dissecting the sampling process, they establish a compelling link between diffusion models and autoregressive decoding, suggesting a generalized bit ordering that transcends traditional coordinate-based approaches. This research pushes the boundaries of diffusion models, establishing their utility for image generation, compression, and potential implications for other machine learning systems.

## Abstract


Diffusion probabilistic models or [Diffusion Models](../permanent/diffusion-models.md), are a class of latent variable models inspired by [Nonequilibrium Thermodynamics](../permanent/nonequilibrium-thermodynamics.md), in this paper, they demonstrate their capability for high-quality image generation.

They get best result by training on a "weighted variational bound" designed according to a novel connection between diffusion probabilistic models and denoising score matching with [Langevin Dynamics](../permanent/langevin-dynamics.md).

The models have a progressive lossy decompression scheme that can be interpreted as a generalisation of autoregressive decoding.

On the unconditional CIFAR10 dataset, they obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17.

On 256x256 LSUN, they obtain sample quality similar to ProgressiveGAN.

Implementation is available at https://github.com/hojonathanho/diffusion.

## Introduction

Deep generative models of all kinds have recently exhibited high quality samples in a wide variety of data modalities.

Generative adversarial networks (GANs), autoregressive models, flows, and variational autoencoders (VAEs) have synthesized striking image and audio samples.

Energy-based modeling and score matching that have produced images comparable to those of GANs.

This paper presents progress in diffusion probabilistic models, which originated in [Deep Unsupervised Learning Using Nonequilibrium Thermodynamics](deep-unsupervised-learning-using-nonequilibrium-thermodynamics.md).

A diffusion probabilistic model (called "diffusion model" for short) is a parameterised [Markov Chain](../permanent/markov-chain.md) trained using variational inference to produce samples matching the data after finite time.

Transitions of the chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed.

When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterisation.

They are straightforward to define and efficient to train.

Capable of generating high quality samples, sometimes better than the published results on other types of generative models (Section 4).

They show certain parameterisation of diffusion models reveals an equivalence with denoising score matching over multiple noise levels during training and with annealed Langevin dynamics during sampling.

We obtained our best sample quality results using this parameterisation, so we consider this equivalence to be one of our primary contributions.

Despite their sample quality, our models do not have competitive log likelihoods compared to other likelihood-based models (our models do, however, have log likelihoods better than the large estimates annealed importance sampling has been reported to produce for energy based models and score matching [11, 55]).

We find that the majority of our models’ lossless codelengths are consumed to describe imperceptible image details (Section 4.3).

We present a more refined analysis of this phenomenon in the language of lossy compression, and we show that the sampling procedure of diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit ordering that vastly generalises what is normally possible with autoregressive models.

## Background

Diffusion models are latent variable models of the form $p_\theta(\mathbf{x}_0) := \int p_\theta(\mathbf{x}_{0:T})d\mathbf{x}_{1:T}$, where $\mathbf{x}_1,\ldots, \mathbf{x}_T$ are latents of the same dimensionality as the data $\mathbf{x}_{0} \sim q(\mathbf{x}_0)$.

The joint distribution $p_\theta(\mathbf{x}_{0:T})$ is called the reverse process, and it is defined as a Markov chain with learned Gaussian transitions starting at $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0},\mathbf{I})$:

$$
p_\theta(\mathbf{x}_{0:T}) := p(\mathbf{x}_T) \prod\limits_{t=1}^{T} p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t), 
\ \ \ \ \ \  p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) := \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_t, t))
$$


What distinguishes diffusion models from other types of latent variable models is the approximate posterior $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$, called the forward process or diffusion process, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_1, \ldots, \beta_T$:

$$
q(\mathbf{x}_{1:T}|\mathbf{x}_0) := \prod\limits_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1}), \ \ \ \ \ \ \ \ \  q(\mathbf{x}_t|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

Training is performed by optimising the usual variational bound on [Negative Log-Likelihood](../permanent/negative-log-likelihood.md):

$$
\mathbb{E}[-\log p_\theta(\mathbf{x}_0)] \leq \mathbb{E}_q\left[-\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right] = \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t\geq1}\log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] := \mathcal{L} \:(3)
$$

The forward process variances $\beta_{t}$ can be learned by reparameterisation or held constant as hyperparameters.





## Experiments

We set $T = 1000$ for all experiments so that the number of neural network evaluations needed during sampling matches previous work [53, 55]. We set the forward process variances to constants increasing linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. These constants were chosen to be small relative to data scaled to $[-1, 1]$, ensuring that reverse and forward processes have approximately the same functional form while keeping the signal-to-noise ratio at $\mathbf{x}_T$ as small as possible ($L_T = D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})) \approx 10^{-5}$ bits per dimension in our experiments).

To represent the reverse process, we use a U-Net backbone similar to an unmasked PixelCNN++ [52, 48] with group normalization throughout [66]. Parameters are shared across time, which is specified to the network using the Transformer sinusoidal position embedding [60]. We use self-attention at the 16 × 16 feature map resolution [63, 60]. Details are in Appendix B.

## Architecture

Backbone: U-Net based on PixelCNN++, replace weight norm with Group Norm.


Our 32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4), and our 256 × 256 models use six.

All models have two convolutional residual blocks per resolution level and self-attention blocks at the 16 × 16 resolution between the convolutional blocks [6].

Diffusion time $t$ is specified by adding the Transformer sinusoidal position embedding [60] into each residual block.

Our CIFAR10 model has 35.7 million parameters, and our LSUN and CelebA-HQ models have 114 million parameters.

We also trained a larger variant of the LSUN Bedroom model with approximately 256 million parameters by increasing filter count.

We used TPU v3-8 (similar to 8 V100 GPUs) for all experiments. Our CIFAR model trains at 21 steps per second at batch size 128 (10.6 hours to train to completion at 800k steps), and sampling a batch of 256 images takes 17 seconds. Our CelebA-HQ/LSUN (256²) models train at 2.2 steps per second at batch size 64, and sampling a batch of 128 images takes 300 seconds. We trained on CelebA-HQ for 0.5M steps, LSUN Bedroom for 2.4M steps, LSUN Cat for 1.8M steps, and LSUN Church for 1.2M steps. The larger LSUN Bedroom model was trained for 1.15M steps.
