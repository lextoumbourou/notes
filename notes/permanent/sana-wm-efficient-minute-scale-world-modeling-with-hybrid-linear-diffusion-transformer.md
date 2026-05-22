---
title: "Sana-WM: 60-second camera-controlled videos on a single GPU"
date: 2026-05-19 00:00
modified: 2026-05-19 00:00
status: draft
tags:
- CameraControlledWorldModelling
paper_title: "Sana-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer"
---

Introduces a new world model called **SANA-WM**, that is camera-controlled and can generate a 60-second clip on a single GPU.
 
The architecture builds on their earlier work with **SANA-Video**

At only 2.6B parameters, they can train it in only 15 days on only 64 H100 GPUs, using on 213K public video clips as training data.

They distill a variant with [NVFP4 Quantization](https://build.nvidia.com/spark/nvfp4-quantization), and demonstrate that it can generate a 60-second clip in 720p in **34 seconds** on a single RTX 5090 GPU. Faster than real time!

  It consists of 4 design choices.
  - Hybrid Linear Attention
  - Dual-Branch Camera Control
  - Two-Stage Generation Pipeline
  - Robust Annotation Pipeline
  
They consider distilling the long-video model from a short-video teacher, but claim that it doesn't provide enoug supervision to give them the scene persistence and "trajectory following" they want.

## Related Work

SANA-WM is a type of **generative simulator** that generates observations given actions or conditions, unlike Representation-centric models like [[JEPA]] and [[I-JEPA]] that learns abstract visual features without generating actual pixels. Generative simulators tend to be more compute-intensive as they have to synthesise every pixel.

Pure [[Softmax Attention]], though it can be accelerated by techniques like [FlashAttention](FlashAttention.md), the memory and compute still grow exponetionall with context length, as every token needs to attend to every other token. And KV cache becomes prohibitively large at the minute scale.

Plücker rays encode 3D lines (the ray from a camera through apxiel) using both its direction vector and [[Moment Vector]] ([Image Moments](../../../permanent/image-moments.md)) - giving a compact per-pixel represetation of exactly where the camrea is and where it's looking. It's a natural way to inject fine-grained camera geometry into a model.

## Method

Progressive Training Strategy

Train progressively from short clips to minute-scale clips, and actually introduce architectural components in each stage.

Stage 1: Efficient VAE Adaption
- replace the baseline VAE with LTX2-VAE to leverage its superior spatiotemporal compression ratio

State 2: Hybrid Architecture Adaptation

To finish.
