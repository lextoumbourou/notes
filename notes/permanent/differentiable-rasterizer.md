---
title: Differentiable Rasterizer
date: 2026-09-26 08:52
modified: 2026-09-26 08:52
status: draft
---

A **Differentiable Rasterizer** renders [Vector Graphics](vector-graphics.md), like the Bézier curves in an SVG, into a pixel image in a way that lets gradients flow back from the image to the shape parameters (control points, colours, stroke widths).

That means vector shapes can be optimised with image-space losses, like a pixel loss or [Score Distillation Sampling](score-distillation-sampling.md). A common implementation is DiffVG, from the paper [Differentiable Vector Graphics Rasterization for Editing and Learning](https://people.csail.mit.edu/tzumao/diffvg/) by Li et al. (2020).

It's used in [NeuralSVG: An Implicit Representation for Text-to-Vector Generation (Jan 2025)](../reference/neuralsvg-an-implicity-represetation-for-text-to-vector-generation-jan-2025.md) to render the predicted shapes so they can be optimised.
