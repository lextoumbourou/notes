---
title: Taming Transformers for High-Resolution Image Synthesis
date: 2024-01-09 00:00
modified: 2026-09-26 08:55
status: draft
---

Notes for paper [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) by Patrick Esser, Robin Rombach, Björn Ommer.

---

This paper introduced VQGAN, which learns a [Codebook](codebook.md) of image constituents. The idea of a learned codebook itself came earlier, from [VQ-VAE](vq-vae.md).

## Abstract

Designed to learn long-range interactions on sequential data, transformers continue to show state-of-the-art results on a wide variety of tasks.

In contrast to CNNs, they contain no inductive bias that prioritizes local interactions. 

This makes them expressive, but also computationally infeasible for long sequences, such as high-resolution images.

We demonstrate how combining the effectiveness of the inductive bias of CNNs with the expressivity of transformers enables them to model and thereby synthesize high-resolution images.

We show how to (i) use CNNs to learn a context-rich vocabulary of image constituents, and in turn (ii) utilize transformers to efficiently model their composition within high-resolution images.

Our approach is readily applied to conditional synthesis tasks, where both non-spatial information, such as object classes, and spatial information, such as segmentations, can control the generated image.

In particular, we present the first results on semantically-guided synthesis of megapixel images with transformers and obtain the state of the art among autoregressive models on class-conditional [ImageNet](ImageNet.md).



