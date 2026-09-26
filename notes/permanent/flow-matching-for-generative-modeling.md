---
title: Flow Matching for Generative Modeling
date: 2024-10-06 00:00
modified: 2026-09-26 08:55
status: draft
---

**Flow Matching for Generative Modeling** is a 2022 paper by Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel and Matt Le ([arXiv](https://arxiv.org/abs/2210.02747)). It introduces Flow Matching, a simulation-free way to train Continuous Normalizing Flows.

Instead of training the flow by simulating an ODE, Flow Matching trains a neural network to regress a vector field that moves samples along a chosen probability path from noise to data. The paths used by [Diffusion Models](diffusion-models.md) are one special case. The paper also proposes optimal transport paths, which are straighter, and found they train faster and need fewer steps to sample.

Flow matching is now used in large image models such as [FLUX.1](flux1.md).

![](../_media/flow-matching-for-generative-modeling-title.png)
![](../_media/flow-matching-for-generative-modeling-title-1.png)

![](../_media/flow-matching-for-generative-modeling-fig-1.png)

![](../_media/flow-matching-for-generative-modeling-fig-3.png)

![](../_media/flow-matching-for-generative-modeling-fig-4.png)
