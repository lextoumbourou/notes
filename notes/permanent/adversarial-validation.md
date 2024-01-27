---
title: Adversarial Validation
date: 2021-05-12 00:00
tags:
  - EDA
---

A technique used to evalute how different the test set is to the training set, commonly used in Kaggle competitions to assess likelihood of a [Leaderboard Shakeup](leaderboard-shakeup.md). It's a form of [Black-box Tests](black-box-tests.md) for datasets.

Steps:

1. Label each element of a dataset `1` if in train or `0` if test.
2. Train a model to predict label using features from the set.

If the model performs well (> 0.5 ROC) then there is likely a difference between datasets.

In [this example](https://www.kaggle.com/lextoumbourou/pp-2020-adversarial-validation), I compared the train and test set of the [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) competition.
