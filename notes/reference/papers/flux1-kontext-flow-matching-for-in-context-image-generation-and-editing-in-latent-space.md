---
title: "FLUX.1 Kontext: Flow Matching for In‑Context Image Generation and Editing in Latent Space"
date: 2025-05-31 00:00
modified: 2025-05-31 00:00
status: draft
---

*My WIP summary of paper [FLUX.1 Kontext: Flow Matching for In‑Context Image Generation and Editing in Latent Space](https://cdn.sanity.io/files/gsvmb6gz/production/880b072208997108f87e5d2729d8a8be481310b5.pdf) by Black Forest Labs*

## Overview

FLUX.1 Kontext is Black Forest Labs’ edit‑oriented text‑to‑image and image‑to‑image model. Starting from a [FLUX.1](../../../../permanent/flux1.md) base checkpoint, they fine‑tune with edit‑focused data and a few small architectural tweaks.

They also introduce **KontextBench**, 1 026 image‑prompt pairs across five tasks (local, global, text, style, character edits) designed to test both single‑turn quality and multi‑turn stability.

## FLUX 1 Kontext

### Objective

Learn
$p_\theta(x \mid y, c)$

* **x** – target image
* **y** – optional reference image
* **c** – text instruction

When **y** is empty it reduces to plain text‑to‑image.

### Key tricks

* **Sequence concat** – Encode *x* and *y* with the frozen FLUX auto‑encoder, then append context tokens after target tokens (channel‑wise concat was worse). Works for arbitrary resolutions and multiple context images.
* **3‑D RoPE offset** – Give all target tokens position \$(0,h,w)\$ and each context image \$y\_i\$ a constant “virtual‑time” offset \$(i,h,w)\$. This cleanly separates context from target with no extra masking.
* **Rectified‑flow loss** – $L_{\theta} = \mathbb{E}_{t,x,y,c} \left\| v_{\theta}(z_t, t, y, c) - (\varepsilon - x) \right\|_2^2$ where $z_t = (1 - t)x + t\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 1)$ and $t$ follows a resolution-shifted logit-normal schedule. Dropping *y* tokens recovers the standard T2I loss.
* **Fast sampling via LADD** – Latent Adversarial Diffusion Distillation halves the step count while improving quality and avoids over‑saturated CFG artifacts.

### Training setup

* Start from a FLUX 1 T2I checkpoint.
* Fine‑tune on millions of ⟨context, image, prompt⟩ triplets.
* Three variants:

  * **pro** – full flow objective → distilled with LADD
  * **dev** – 12 B diffusion transformer distilled from **pro**, image‑to‑image only
  * **max** – like **pro** but with extra compute

### Safety & infra

* CLIP‑style classifier filtering plus adversarial training against NCII/CSEM.
* FSDP‑2 mixed‑precision (bf16 gather / fp32 reduce), selective activation checkpointing, Flash‑Attention 3, region‑wise compilation.

## Results

### Text‑to‑image

On KontextBench, Recraft v3 tops **Aesthetics** and **Realism**. GPT‑Image (high) rules **Prompt Following** and **Typography**, matching my notes in [Imagen 4 is faster, but GPT is still the GOAT](../../permanent/imagen4-is-faster-but-gpt-is-still-the-goat.md).

FLUX.1 Kontext is a near‑top contender in every metric despite not being tuned purely for T2I.

### Image‑to‑image editing

Kontext dominates all five editing tasks, especially multi‑turn consistency.

See my own test: [Consistent Doggo with Flux.1: Kontext](../../permanent/consistent-doggo-with-flux-1-kontext.md).
