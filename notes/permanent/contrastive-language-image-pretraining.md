---
title: CLIP
date: 2023-12-06 00:00
modified: 2023-12-06 00:00
aliases:
  -  Contrastive Language-Image Pretraining (CLIP)
summary: a model that can associate textual representations with images.
---

**Contrastive Language-Image Pretraining** or **CLIP** is an approach to training a model to associate images with their textual representations using [Contrastive Loss](contrastive-loss.md). This allows for high-performing [Zero-Shot Learning](../public/notes/permanent/zero-shot-learning.md) i.e. the model can generalise to new tasks without fine-tuning.

The architecture is a "simplified version of [ConVIRT](convirt.md)" [^1] trained from scratch.

From paper [Learning Transferable Visual Models From Natural Language Supervision](../reference/learning-transferable-visual-models-from-natural-language-supervision.md)

[^1]: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020. https://arxiv.org/abs/2103.00020
