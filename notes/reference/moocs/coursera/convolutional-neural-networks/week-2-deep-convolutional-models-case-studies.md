---
title: "Week 2 - Deep convolutional models: case studies"
date: 2017-11-29 00:00
category: reference/moocs
parent: convolutional-neural-networks 
status: draft
---

## Outline

* Reading papers
* Classic networks:
  * LeNet-5
  * AlexNet
  * VGG
  * ResNet
  * Inception

## Classic networks

### LeNet-5

* Trained using 32x32x1 (grayscale) images.
* No padding, width and height of filters shrinks throughout network but num channels increases.
* Uses sigmoid and tanh activations - before the time of relu.

### AlexNet

* Trained on 227x227x3 images.
* First network to really sell people on deep learning's potential.

### VGG-16

* Simpler approach.
* Uses less hyperparams.
* Make use of padding and doubles conv size on each layer.

### ResNet

* Solves problem of exploding or vanishing gradients on very deep networks.
* Uses a "residual block", which feeds input data into later neurons, allowing a network to easily learn the identity function for unimportant paths.

## Networks in Networks and 1x1 Convolutions

### Inception
