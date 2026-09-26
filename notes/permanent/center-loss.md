---
title: Center Loss
date: 2026-09-26 08:50
modified: 2026-09-26 08:50
status: draft
---

**Center Loss** is a [Loss Function](loss-function.md) for learning discriminative features, introduced in [A Discriminative Feature Learning Approach for Deep Face Recognition](../reference/papers/a-discriminative-feature-learning-approach-for-deep-face-recognition.md).

It learns a center (a vector of the same dimension as the feature embedding) for each class, and penalises the distance between each feature and its class center:

$$
L_{C} = \frac{1}{2} \sum\limits_{i=1}^{m} {||\mathbf{x}_i - \mathbf{c}_{y_i}||}^{2}_{2}
$$

Where $m$ is the mini-batch size and $\mathbf{c}_{y_i}$ is the center of the class $y_i$. This pulls features of the same class close together. It's trained jointly with [Softmax Loss](softmax-loss.md), which keeps features of different classes apart, and a hyperparameter $\lambda$ balances the two. On its own, the centers and features would collapse to zero, since that gives the lowest possible center loss.
