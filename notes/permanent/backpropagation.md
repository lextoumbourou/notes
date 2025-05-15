---
title: Backpropagation
date: 2024-10-07 00:00
modified: 2024-10-07 00:00
status: draft
---

**Backpropagation** is an algorithm used to train neural networks, where it calculates the gradient of a loss function concerning each weight by propagating the error backward from output layer to input layer. The process enables the network to minimise error and improve predictive accuracy over time.

First described in [Learning Representations by Back-propagating Errors](../../../reference/learning-representations-by-back-propagating-errors.md).

Each step of back-propagation first performs a forward-propagation of the input signal to generate a prediction, then compares the prediction to a desired target, and finally propagates the error signal back through the network to determine how the weights of each layer should be adjusted to decrease the error.