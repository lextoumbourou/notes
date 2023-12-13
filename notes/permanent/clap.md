---
title: CLAP
date: 2023-12-13 10:00
modified: 2023-12-13 10:00
aliases:
  - Contrastive Language-Audio Pre-training
cover: /_media/cover-clap-paper.png
summary: A pre-training system for learning a text/audio latent space
tags:
  - MachineLearning
  - AudioEngineering
  - UnsupervisedLearning
---

CLAP, Contrastive Language-Audio Pre-training, is a pre-training system for learning a text/audio latent space given pairs of examples, inspired by [CLIP](https://openai.com/research/clip). It is trained using [Contrastive Loss](contrastive-loss.md), hence: <span style="color: red;">Contrastive</span> <span style="color: blue;">Language-Audio</span> <span style="color: green;">Pre-training</span>.

They use a text-encoder and audio-encoder to generate respective representations, then feed into an neural network to learn a shared latent space. For the text-encoder, they use [RoBERTa](../../permanent/RoBERTa.md), and for the audio-encoder, they use [HTS-AT](../../permanent/hts-at.md), after experimenting with various alternatives (see paper).

The audio/text representations can be used for [Text-to-Audio Retrieval](text-to-audio-retrieval.md), [Zero-shot Classification](zero-shot-classification.md), or fine-tuned downstream for supervised classification.

They released the architecture, training code and series of weights trained on different subsets of their datasets [here](https://github.com/LAION-AI/CLAP).

The system and architecture are described in the paper, [Large-scale Contrastive Language-Audio Pre-training with Feature Fusion and Keyword-to-Caption Augmentation](../reference/papers/large-scale-contrastive-language-audio-pre-training-with-feature-fusion-and-keyword-to-caption-augmentation.md).