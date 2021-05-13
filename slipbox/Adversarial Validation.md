Title: Adversarial Validation
Date: 2021-05-12
Tags: #MachineLearning #KaggleTricks

---

A technique used to evalute how different the test set is to the training set.

Steps:
1. Label each element of a dataset: `is_train=True/False`.
2. Train a model to predict label.

If the model performs well then there is likely a significant different between datasets.

Commonly used in Kaggle competitions to assess the possibility of a leaderboard shakeup.

In [this example](https://www.kaggle.com/lextoumbourou/pp-2020-adversarial-validation), I compared the train and test set of the [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) competition.

In a way, this is a machine learning equivalent of [[Black-box testing]].