---
title: Denoising Diffusion Model
date: 2024-01-09 00:00
modified: 2026-09-26 08:50
status: draft
---

A type of [Generative Model](generative-model.md) used in Machine Learning.

How it works:

1. Start with noise: model begins with random noise image: a random collection of pixels.
2. Gradual refinement: gradually transform this noise into a coherent image over a series of steps. At each step, the model attempts to "denoise" or reduce the randomness of the image. This is done using a deep learning network that has learned how to progressively add structure and detail.
3. Learning Process: during training, the model learns by doing the opposite of the generative process. Starts with real images and learns how to gradually add noise to them, doing the process in reverse. The model learns about underlying structure of the image data.
4. Controlled Generation: in generation phase, the model applies the learned transformations in reverse. Starting from noise, it incrementally removes that noise and adds detail, following the learned patterns to create an image. The process is guided by a learned probability distribution that dictates what kinds of images are likely to be generated from the noise.
5. Denoising diffusion models can be conditioned on various inputs, like text description, for specific types of images.
