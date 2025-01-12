---
title: "DreamFusion: Text-to-3D Using 2D Diffusion"
date: 2025-01-11 00:00
modified: 2025-01-11 00:00
status: draft
---

*These are my notes from the paper [DreamFusion: Text-to-3D Using 2D Diffusion](https://arxiv.org/abs/2209.14988) (September 2022) by Poole, Jain, Barron, and Mildenhall*

### Overview

- First text-to-3D model that requires no 3D training data
- Uses pre-trained 2D text-to-image diffusion model (Imagen) to guide 3D synthesis
- Introduces [Score Distillation Sampling](../../permanent/score-distillation-sampling.md) (SDS) - a new technique to optimize using diffusion models
- Creates high-quality 3D models that can be viewed from any angle and relit

### Key Details

- Neural Radiance Field (NeRF) optimization using 2D diffusion model guidance
- View-dependent prompting to ensure consistency across angles
- Textureless rendering during training to enforce proper 3D geometry
- No modification needed to the underlying text-to-image model

> [!question] What is the key innovation that allows DreamFusion to work without 3D training data?
> a) Using a specialized 3D architecture
> b) Training on synthetic 3D data
> c) Using a 2D diffusion model as a differentiable loss
> d) Using multiple camera views during training
> > [!success]- Answer
> > c) Using a 2D diffusion model as a differentiable loss
> > The paper's key innovation is using Score Distillation Sampling to turn a 2D diffusion model into a loss function for optimizing 3D content.

### Background

- Previous text-to-3D methods required:
    - Large 3D training datasets
    - Category-specific training (e.g., only faces)
    - Or used CLIP which gave lower quality results
- Builds upon:
    - NeRF for 3D representation
    - Imagen for text-to-image guidance
    - Dream Fields for optimization-based 3D synthesis

> [!question] What are the limitations of previous text-to-3D approaches that DreamFusion addresses?
> a) High computational cost
> b) Need for 3D training data
> c) Limited to specific object categories
> d) Both b and c
> > [!success]- Answer
> > d) Both b and c
> > Previous methods either needed large 3D datasets or only worked on specific categories like faces.

### Method
1. Score Distillation Sampling:
    - Converts diffusion model into a differentiable loss
    - Optimizes parameters directly without backpropagating through the model
    - Uses classifier-free guidance with high guidance weights (ω=100)

2. NeRF Architecture:
    - Represents scene within bounding sphere
    - Uses integrated positional encoding
    - Outputs density and RGB albedo
    - Includes diffuse lighting model
    - Background handled by separate MLP

3. Training Process:
    - Random camera and lighting positions
    - Mixture of textured and textureless renders
    - View-dependent prompt conditioning
    - Geometry regularization to prevent degenerate solutions

> [!question] Why does DreamFusion use textureless rendering during training?
> a) To save computational resources
> b) To prevent the model from "painting" 2D images on flat surfaces
> c) To improve rendering speed
> d) To reduce memory usage
> > [!success]- Answer
> > b) To prevent the model from "painting" 2D images on flat surfaces
> > Textureless rendering forces the model to create proper 3D geometry rather than finding shortcuts by drawing 2D content on flat surfaces.

### Experiments

- Evaluated on:
    - 153 prompts from object-centric COCO validation subset
    - CLIP R-Precision metric for both textured and textureless renders
- Compared against:
    - Dream Fields
    - CLIP-Mesh
    - Ground truth MS-COCO images
- Results:
    - Outperforms baselines on colored renders
    - Significant improvement in geometric accuracy (58.5% vs ~1% for baselines)
    - Maintains consistency across views
    - Works on diverse range of subjects and styles

> [!question] How does DreamFusion evaluate geometric quality?
> a) Using 3D scanning
> b) Manual inspection
> c) CLIP R-Precision on textureless renders
> d) Direct comparison with ground truth 3D models
> > [!success]- Answer
> > c) CLIP R-Precision on textureless renders
> > The paper evaluates geometric quality by measuring CLIP R-Precision on textureless renders, which can only match the text if the underlying geometry is correct.

[Figure 1 should be shown here - Main results figure showing various 3D models generated from text]
[Figure 3 should be shown here - System architecture diagram]
[Figure 4 should be shown here - Progressive refinement example]
[Figure 6 should be shown here - Ablation study results]