---
title: Segment Anything
date: 2024-10-11 00:00
modified: 2024-10-11 00:00
status: draft
---

Notes from paper [Segment Anything](https://arxiv.org/abs/2304.02643v1) by Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick.

## Overview

The project aims to build a foundation model for [Image Segmentation](../../permanent/image-segmentation.md), capable of performing zero-shot transfer to new image distributions and tasks.

The authors introduce three interconnected components:
* a prompt-able segmentation task
* a segmentation model called [Segment Anything Model](../../permanent/segment-anything-model.md)
* a data engine for collecting [SA-1B Dataset](../../permanent/sa-1b-dataset.md), a dataset of over one billion masks.

The paper talks about the importance of [Prompt Engineering](../../../../permanent/prompt-engineering.md) to achieve [Zero-Shot Transfer](../../permanent/zero-shot-transfer.md).
