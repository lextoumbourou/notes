---
title: "Quattoni et al., 2007: Learning Visual Representations using Images with Captions"
date: 2024-10-13 00:00
modified: 2026-09-26 08:55
status: draft
---

**Learning Visual Representations using Images with Captions** is a 2007 CVPR paper by Ariadna Quattoni, Michael Collins and Trevor Darrell.

It's an early example of using natural language as supervision for computer vision. The paper trains classifiers to predict words in the captions that come with images, then uses [Manifold Learning](../permanent/manifold-learning.md) in the weight space of those classifiers to learn image representations that make it more data-efficient to learn new visual categories.

It's cited as related work in [Learning Transferable Visual Models From Natural Language Supervision](learning-transferable-visual-models-from-natural-language-supervision.md) (CLIP).
