---
title: Presizing
date: 2021-08-05 00:00
tags:
  - MachineLearning
  - ImageClassification
cover: /_media/presizing-cover.png
---

Pre-sizing is a technique for preparing images for training a neural network.

In standard image pipelines, images are resized to a size suitable for our network (i.e., 224x224) and then augmented. This process results in empty regions and degraded data.

If we first resize the images to some smaller size that is well above our target (i.e., 460x460), we are left with a spare margin to perform augmentations and our target-sized crop without empty regions.

[@howardDeepLearningCoders2020] *(pg. 217-219)*

Without presizing (using the [imgaug](https://imgaug.readthedocs.io/en/latest/) library):

![Presizing](_media/presizing-standard-aug.png)

With presizing (using the [fastai library](https://docs.fast.ai)):

![Presizing](_media/presizing-cover.png)

These examples use an image from the [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) dataset. You can see the source [here](https://www.kaggle.com/lextoumbourou/presizing).

The [fastai library](https://docs.fast.ai) also combines augmentations into a single operation, minimizing the number of lossy operations performed. Additionally, fastai can perform augmentation operations on the GPU.

The concept of resizing images before training is common practice when dealing with large source image sizes. Especially when training in environments with smaller disk allocations like Google Collab. The smaller images afford faster experiments and more loops, which is as crucial in Machine Learning as in Game Design (see [Rule of Loop](rule-of-loop.md) ).
