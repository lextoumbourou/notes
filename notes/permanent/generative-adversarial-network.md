---
title: Generative Adversarial Network
date: 2024-03-14 00:00
modified: 2026-09-26 08:50
status: draft
---

A **Generative Adversarial Network** (GAN) is a type of [Generative Model](generative-model.md) made up of two neural networks trained against each other: a generator, which turns random noise into fake samples, and a discriminator, which tries to tell real samples from generated ones.

The generator is trained to fool the discriminator, while the discriminator is trained to catch it out, so over time the generated samples get more realistic. GANs were introduced by Ian Goodfellow and colleagues in 2014, and were the leading approach for image generation until [Diffusion Models](diffusion-models.md) caught up with them. The adversarial idea is still widely used as a training loss in other models, like [VQ-GAN](vq-gan.md) and [HiFi-GAN](hifigan.md).
