---
title: Overfit First
date: 2024-01-21 00:00
modified: 2024-01-21 00:00
cover: /_media/overfit-first.png
summary: the first step to training a neural network successfully
tags:
- MachineLearning
- SoftwareEngineering
---

In a [Tweet by Karpathy from 2019](https://twitter.com/karpathy/status/1013244313327681536?lang=en), he mentions his #1 neural network mistake: *"You didn't try to overfit a single batch first"*. Jeremy Howard shares a similar sentiment in his [Three Steps to Training a Good Model](https://www.youtube.com/watch?v=4u8FxNEDUeg&t=1267s) from the same year, where he lists the *Overfit* as the #1 step:

![Three Steps to Training a Good Model by Jeremy Howard](../_media/overfit-first-3-steps.png)
Slide from this [fastai lesson](https://www.youtube.com/watch?v=4u8FxNEDUeg&t=1267s0).

In my experience, failing to check that the model can overfit a small amount of data is one of the surest ways to waste time in machine learning. If the model cannot learn to perfectly perform the task on the data it was trained on, then what hope do you have training on a bigger dataset?

Also, by starting with the overfitting, you allow the data collection and model construction to happen in parallel, which is much closer to the goal of the [Iterative Development](iterative-development.md) goals of the Agile software methodology. Instead of spending time collecting data upfront, it can be done alongside the model development, allowing for parallel progress streams throughout the project. It also means each step can inform each other: model results can be used to assess the data most pertinent for modelling the problem, and available data can tell what model architectures make the most sense for the project.

Always start a neural network training endeavour by testing that you can overfit on a small amount of data first.