---
title: Presizing
date: 2021-08-05 18:00
tags:
  - DeepLearning
  - ImageClassification
cover: /_media/presizing-cover.png
---

Pre-sizing is a technique for preparing images for training a network.

In standard image pipelines, we resize images to a size suitable for our network (i.e., 224x224) and then perform augmentations. This process results in degraded data and "spurious empty zones."


If we instead resize the images to some smaller size that is well above our target (i.e., 460x460), then perform augmentations and then finally resize to our target image size, we can maximize our image quality.

[@howardDeepLearningCoders2020] *(pg. 217-219)*

This example performs pre-sizing using the fastai library on a leaf from the [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) dataset.

Without presizing:

![Presizing](/_media/presizing-standard-aug.png)

With presizing:

![Presizing](/_media/presizing-cover.png)

The fastai library first combines augmentations into a single operation which minimizes the number of lossy operations performed. Additionally, fastai can perform augmentation operations on the GPU.

The concept of resizing images before training is common practice, especially when dealing with large source image sizes. Especially when training in environments with smaller disk allocations like Google Collab. The smaller images afford faster experiments and more loops, which is as important in Machine Learning as in Game Design (see [[Rule of Loop]]).