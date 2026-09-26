---
title: Rectified Flows
date: 2026-05-24 00:00
modified: 2026-09-26 08:55
status: draft
---

**Rectified Flows** are a generative modelling approach that learns an ODE to transport samples from a noise distribution to the data distribution along paths that are as straight as possible. The model is trained to predict the velocity pointing from noise to data.

Straight paths mean you can generate samples in far fewer steps than a typical [Diffusion Models](diffusion-models.md). It's closely related to [Flow Matching](flow-matching.md).

Used in [SANA-Video](SANA-Video.md).
