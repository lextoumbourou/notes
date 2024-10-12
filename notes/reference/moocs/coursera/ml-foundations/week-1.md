---
title: Week 1 - Welcome
date: 2015-10-07 00:00
category: reference/moocs
status: draft
parent: ml-foundations
---

## Course case studies

* Case Study 1: Predicting house prices
    * Data:
        * prices of other houses.
        * other features of houses (number of bedrooms etc).
    * ML method: [Regression](../../../../permanent/regression.md)
* Case Study 2: Sentiment analysis
    * Data:
        * look at other reviews.
        * "decision boundary" -on
    * ML method: [Classification](../../../../permanent/classification.md)
* Case Study 3: Document retrieval
    * Data:
        * collection of articles.
    * ML Method: [Clustering](../../../../permanent/clustering.md)
* Case Study 4: Product recommendation
    * Data:
        * past purchases
    * ML Method: [Matrix Factorization](Matrix%20Factorization)
* Case Study 5: Visual product recommender
    * Data:
        * input images (shoes)
    * ML Method: [Deep Learning](Deep%20Learning)

## 2. Regression

* Linear regression
* Regularization:
    * Ridge (L2)
    * [Lasso](../../../../permanent/lasso.md) (L1)
* Algorithms
    * Gradient descent
    * Coordinate descent
* Concepts
    * Loss functions
    * Bias-variance tradeoff
    * Cross-validation
    * Sparsity
    * Overfitting
    * Model selection

## 3. Classification

* Models
    * Linear classifiers
    * Kernels
    * Decision trees
* Algorithms
    * Stochastic gradient descent
    * Boosting
* Concepts
    * Decision boundaries
    * MLE
    * Ensemble methods
    * Random forest
    * CART
    * Online learning

## 4. Clustering & Retrieval

* Models
    * Nearest neighbors
    * Clustering, mixtures of Gaussians
    * Latent Dirichlet allocation (LDA)
* Algorithms
    * KD-trees, locality-sensitive hash (LSH)
    * K-means
    * Expectation-maximization (EM)
* Concepts
    * Distance metrics
    * Approximation algorithms
    * Hashing
    * Sampling algorithms
    * Scaling up with map-reduce

## 5. Matrix factorizatoin & Dimensionality Reduction

* Models
    * Collaborative filtering
    * Matrix factorization
    * [Principal Component Analysis](../../../../permanent/principal-component-analysis.md)
* Algorithms
    * Coordinate descent
    * Eigen decomposition
    * SVD
* Concepts
    * Matrix completion
    * Eigenvalues
    * Random projects
    * Cold-start problem
    * Diversity
    * Scaling up

## 6. Capstone: intelligent application using deep learning

* Build & deploy a recommender using product images and text sentiment.

## Goals of course

* Minimum prereqs required to understand the concepts
* Target audience
    * Software engs
    * Data enthusiast
    * Scientist
* Math background
    * Basic calculus (lol)
        * Concept of derivatives
    * Basic linear algebra
        * Vectors
        * Matrices
        * Matrix multiplies
* Programming experience
    * Basic Python

## Notes about Graphlab

* Loading data set (somewhat similiar to Panda's dataframe, I guess?)

          import graphlab
          sf = graphlab.SFrame('people-example.csv')

* Basic operations

```
>>> # View dataset
>>> sf
+------------+-----------+---------------+-----+
| First Name | Last Name |    Country    | age |
+------------+-----------+---------------+-----+
|    Bob     |   Smith   | United States |  24 |
|   Alice    |  Williams |     Canada    |  23 |
|  Malcolm   |    Jone   |    England    |  22 |
|   Felix    |   Brown   |      USA      |  23 |
|    Alex    |   Cooper  |     Poland    |  23 |
|    Tod     |  Campbell |      USA      |  22 |
|   Derek    |    Ward   |  Switzerland  |  25 |
+------------+-----------+---------------+-----+
[7 rows x 4 columns]

>>> # Make histogram of ages (open in browser)
>>> sf['age'].show(view='Categorical')

>>> # Get mean age
>>> sf['age'].mean()

>>> # Apply transform function to data set
>>> transform_country = (
      lambda country: 'United States' if country == 'USA' else country_
>>> sf['Country'] = sf['Country'].apply(transform_country)
```
