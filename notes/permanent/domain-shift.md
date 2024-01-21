---
title: Domain Shift
date: 2021-06-26 00:00
tags:
  - MachineLearning
  - MachineLearningFailureModes
cover: /_media/old-vs-new-vans.jpg
summary: When production data diverges significantly from the training dataset
---

When the data your model sees has diverged significantly from the training dataset, it's called a Domain Shift.

[@howardDeepLearningCoders2020] *(pg. 104)*

In the [2019 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2019), PBS Kids ran a Kaggle competition to determine how well a player would do on a challenge based on their behavior so far. Because the training data was particular to the current version of the game, including very level-specific data like the coordinates of mouse clicks, the models would be sensitive to tiny changes made to the game. Changing the order of levels or moving a sprite could trigger a significant domain shift, making the current production data effectively [Out-of-Domain](out-of-domain-data.md).

Cover [New Vans Vs. Old Vans by Danny Lopez on Flickr](https://www.flickr.com/photos/danny24valve/14670135259).
