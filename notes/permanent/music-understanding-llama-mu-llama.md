---
title: Music Understanding LLaMA (MU-LLaMA)
date: 2023-10-15 00:00
modified: 2026-09-26 08:55
status: draft
---

**Music Understanding LLaMA (MU-LLaMA)** is a model that can answer questions about a piece of music and write captions for it.

It uses a frozen, pretrained MERT model to extract music features, then feeds them into the LLaMA language model. Since there weren't many suitable datasets, the authors generated question-answer pairs from existing music captioning datasets to create the MusicQA dataset.

From paper [Paper summary: Music Undersanding LLAMA: advancing text-to-music generation with question answering and captioning](../reference/papers/music-understanding-llama.md).
