---
title: Presizing
date: 2021-08-05 18:00
tags:
  - DeepLearning
  - ImageClassification
cover: /_media/presizing-cover.png
---

Pre-sizing is a technique for preparing images for training a neural network.

In standard image pipelines, images are resized to a size suitable for our network (i.e., 224x224) and then augmented. This process results in empty regions and degraded data.

If we instead resize the images to some smaller size that is well above our target (i.e., 460x460), then perform augmentations including a target-sized crop, this gives us "spare margin," allowing for transforms without empty regions.

[@howardDeepLearningCoders2020] *(pg. 217-219)*

This example performs pre-sizing using the fastai library on a leaf from the [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) dataset.

Without presizing:

![Presizing](/_media/presizing-standard-aug.png)

With presizing:

![Presizing](/_media/presizing-cover.png)

The fastai library also combines augmentations into a single operation which minimizes the number of lossy operations performed. Additionally, fastai can perform augmentation operations on the GPU.

The concept of resizing images before training is common practice when dealing with large source image sizes. Especially when training in environments with smaller disk allocations like Google Collab. The smaller images afford faster experiments and more loops, which is as crucial in Machine Learning as in Game Design (see [[Rule of Loop]]).