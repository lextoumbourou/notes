---
title: Score Distillation Sampling
date: 2025-01-11 00:00
modified: 2025-01-11 00:00
status: draft
---

**Score Distillation Sampling** (SDS) represents a fundamental breakthrough in how we can use diffusion models beyond just generating images. At its core, SDS transforms a diffusion model into a differentiable loss function that can guide optimisation of any differentiable generator. It was introduced in the [DreamFusion: Text-to-3D Using 2D Diffusion](../reference/papers/dreamfusion-text-to-3d-using-2D-diffusion.md) paper.

## SDS vs Traditional Diffusion Sampling

In standard diffusion models, we generate images through a process called ancestral sampling. This involves starting with pure noise and gradually denoising it using many small steps. Each step requires running the diffusion model's U-Net to predict the noise that was added, then removing a portion of that noise. While this works well for generating images directly, it's not suitable for optimising parameters of a generator.

The breakthrough of SDS comes from realising that we don't need to follow the sampling process. Instead, we can use the diffusion model's learned understanding of image structure to define a loss function. This loss function measures how well any generated image matches the model's learned patterns at various noise levels.

> [!question] What is the fundamental difference between SDS and traditional diffusion sampling?
> a) SDS is faster than traditional sampling
> b) SDS transforms the diffusion model into a loss function for optimization
> c) SDS produces higher quality images
> d) SDS requires less memory
> > [!success]- Answer
> > b) SDS transforms the diffusion model into a loss function for optimization
> > This is the key innovation that enables using diffusion models to optimize any differentiable generator.

> [!question] Why isn't traditional ancestral sampling suitable for parameter optimization?
> a) It's too slow
> b) The sampling process isn't differentiable
> c) It requires too much memory
> d) It only works with images
> > [!success]- Answer
> > b) The sampling process isn't differentiable
> > The discrete steps in ancestral sampling make it impossible to backpropagate through the process for optimization.

## The Mathematics Behind SDS

The SDS loss is based on probability density distillation. For any generated image, we add varying amounts of noise and ask the diffusion model to predict what noise was added. The loss is computed as the difference between the predicted noise and the actual noise we added. Mathematically, this looks like:

$$LSDS = Et,ε[w(t) * (εhat(zt|y,t) - ε) * ∂x/∂θ]$$

where:
- t is a randomly sampled timestep
- ε is random noise
- εhat is the model's noise prediction
- w(t) is a weighting function
- ∂x/∂θ represents how the generated image changes with respect to the parameters

> [!question] In the SDS loss formula, what does εhat - ε represent?
> a) The total amount of noise in the image
> b) The difference between predicted and actual noise
> c) The gradient of the loss function
> d) The weighting function
> > [!success]- Answer
> > b) The difference between predicted and actual noise
> > This difference provides the direction for updating the generator's parameters.

## The Implementation Details

A crucial detail is that we don't backpropagate through the diffusion model itself. Instead, we treat the difference between predicted and actual noise as a fixed gradient direction. This makes optimization much more efficient and stable. The gradient tells us how to adjust our generator's parameters to make its outputs more like what the diffusion model expects to see.

## Why It Works Better

Previous approaches tried using the diffusion training loss directly, but this required backpropagating through the U-Net and was unstable. SDS avoids this by using the diffusion model as a critic that provides direct update directions. The use of multiple noise levels also helps provide more robust guidance than single-scale losses like those used in CLIP-based methods.

> [!question] What advantage does using multiple noise levels provide in SDS?
> a) Faster training
> b) Lower memory usage
> c) More robust guidance across scales
> d) Better image quality
 > > [!success]- Answer
 > > c) More robust guidance across scales
 > > Multiple noise levels help provide guidance about both fine details and overall structure.

## The Practical Impact

In DreamFusion, SDS enables the optimization of a NeRF model using only a frozen 2D diffusion model as guidance. The loss provides meaningful gradients that help shape both the geometry and appearance of the 3D scene. By rendering from multiple viewpoints and using SDS as the loss, the NeRF learns to create consistent 3D objects that look good from any angle.

## Stability and Tuning

SDS requires careful tuning of a few key parameters. The guidance scale (how much to amplify the conditional versus unconditional gradients) typically needs to be much higher than in standard diffusion sampling - often around 100 rather than 7-8. The weighting function w(t) across noise levels also impacts results, though a simple schedule often works well.

> [!question] Why does SDS typically use much higher guidance scales (around 100) compared to standard diffusion sampling (7-8)?
> a) To compensate for the lack of training data
> b) To speed up convergence
> c) Because of its mode-seeking nature in optimization
> d) To improve image quality
> > [!success]- Answer
> > c) Because of its mode-seeking nature in optimization
> > The higher guidance scale helps prevent the optimization from finding degenerate solutions.

## Beyond Text-to-3D

While DreamFusion uses SDS for text-to-3D synthesis, the technique is more general. It can be used to optimize any differentiable generator using a diffusion model as guidance. This could enable new applications in domains like audio synthesis, video generation, or even scientific applications where we want to optimize parameters to match complex learned patterns.

The introduction of SDS represents a fundamental advance in how we can use diffusion models. Rather than just sampling from them directly, we can use them as sophisticated critics to guide optimization in new domains. This opens up exciting possibilities for generating complex content while leveraging the powerful knowledge captured in pre-trained diffusion models.