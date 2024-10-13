---
title: Grokking at the Edge of Linear Separability
date: 2024-10-14 00:00
modified: 2024-10-14 00:00
status: draft
tags:
- MachineLearning
---

These are my incomplete notes for the paper [Grokking at the Edge of Linear Separability](https://arxiv.org/abs/2410.04489) by Alon Beck, Noam Levi, and Yohai Bar-Sinai.

## Overview

This research paper investigates the phenomenon of [Grokking](../../permanent/grokking.md) in machine learning, where a model initially memorises training data without generalising to unseen data but later achieves generalisation.

The authors study the generalisation properties of binary [Logistic Regression](../../permanent/logistic-regression.md) in a simplified setting, for which *memorising* and *generalising* solutions are strictly defined, allowing them to understand the mechanism underlying Grokking in its dynamics.

They demonstrate that grokking is amplified when the training data is close to the point of linear separability, and they provide a theoretical framework to explain this observation.

The paper draws parallels between grokking and "critical slowing down" in physics, suggesting a deeper connection between emergent phenomena in machine learning and the intrinsic properties of data.

The work is divided into two parts:
* First, they studied the dynamics of "noise" classification and demonstrated that grokking is exhibited when the classifier transitions from the "memorising" solution to the "generalising" one. Additionally, they show that this grokking transition strongly depends on the underlying properties of the data, as it only occurs when the data is on the verge of being linearly separable from the origin in d dimensions, given by $\lambda c$.
* In the second part, they presented a simple, one-dimensional, effective description of the full problem and showed that it captures the salient properties leading to grokking. Employing this simple model, we can show that the grokking time diverges as the data becomes closer to linearly separable in d dimensions and provide analytical predictions for the grokking time as a function of the properties of the data alone.

The existence of $\lambda c$, which can be identified as the effective interpolation threshold for this problem, shows a potential connection between grokking and other emergent phenomena in deep learning, such as double descent.

They hope to extend the results to more complex data in future research.
