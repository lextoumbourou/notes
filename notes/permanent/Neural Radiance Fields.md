---
title: Neural Radiance Fields
date: 2025-01-10 00:00
modified: 2026-09-26 08:55
status: draft
---

**Neural Radiance Fields** (NeRF) represent a 3D scene as a neural network. The network takes a 3D position and a viewing direction as input and outputs a colour and a volume density for that point.

To render a new view of the scene, you cast a ray through each pixel, sample points along the ray, query the network at each point and combine the results with volume rendering. Because rendering is differentiable, the network can be trained from a set of photos of the scene with known camera positions.

See the paper that introduced the idea: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al., 2020).
