---
title: U-Net
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**U-Net** is a [Convolutional Neural Network](convolutional-neural-network.md) architecture for [Image Segmentation](image-segmentation.md), introduced in [U-Net: Convolutional Networks for Biomedical Image Segmentation](../reference/papers/u-net-convolutional-networks-for-biomedical-image-segmentation.md) (Ronneberger, Fischer and Brox, 2015).

It has a contracting path (encoder) that downsamples the image to capture context, and an expanding path (decoder) that upsamples back to full resolution for precise localisation. Skip connections copy feature maps from each encoder level to the matching decoder level. Drawn out, the architecture looks like a U, hence the name.

It was designed to work well with very few training images, which is common in biomedical imaging. U-Nets were later adopted as the backbone of many [Diffusion Models](diffusion-models.md).
