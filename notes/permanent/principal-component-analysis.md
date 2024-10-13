---
title: Principal Component Analysis
date: 2015-09-08 00:00
modified: 2024-04-10 00:00
summary: a technique for reducing the dimensionality of data
cover: /_media/pca-cover.png
hide_cover_in_article: true
tags:
  - MachineLearning
  - DataScience
---

**Principal Component Analysis**, or **PCA**, is an unsupervised learning technique for reducing the dimensionality of data, used for data visualisation, noise reduction and as preprocessing for machine learning algorithms.

Given a data set, we normalise it by subtracting the mean and dividing it by the standard deviation, then calculating a covariance matrix.

Next, we calculate the [Eigenvector](eigenvector.md) and [Eigenvalue](eigenvalue.md) for the covariance matrix. In PCA, the eigenvectors represent the principal components, which are the directions of maximum variance in the data. The eigenvalues, on the other hand, indicate the amount of variance explained by each principal component.

The eigenvectors are sorted in descending order based on their eigenvalues, representing the variance explained by each principal component. We fetch the top $N$ Eigenvectors by Eigenvalues (where N is the number of components).

Then, a dot product with the normalised data and the N Eigenvectors is taken, which gives the transformed data in lower-dimensional space.

Here's a complete example using the Iris dataset.

{% notebook permanent/notebooks/pca-example.ipynb %}
