---
title: Out-of-Domain
date: 2021-06-22 00:00
cover: /_media/odd-one-out.png
tags:
  - MachineLearning
  - MachineLearningFailureModes
---

When data is provided to a model that is significantly different from what it was trained on, it's referred to as out-of-domain data.

 [@howardDeepLearningCoders2020] *(pg. 104)*

At one company, we trained a model to infer the topics of social media posts using a dataset we painstaking labeled by hand. However, when we set the model loose on a firehose of data, we learned that much social media data text isn't English. A model trained exclusively on English text will have no ability to classify the topics of Japanese writing, for example.

The solution was first to filter the data fed into the model through a language classifier, allowing only English content to be classified by the model until we had the skills to expand into other languages.

What's more, a model trained to classify the content of blogs performs very poorly on shorter form style of posts like those on Instagram. The dataset needs to be carefully balanced to include all types of data the model sees in production.

Classifying datasets is the idea behind [Adversarial Validation](adversarial-validation.md), which uses models to classify the closeness of domains between 2 datasets.

Cover image by [rawdonfox on Flickr](https://www.flickr.com/photos/34739556@N04/6802867364).
