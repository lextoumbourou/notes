# Week 1

## Settings up your Machine Learning application

### Train / Dev / Test sets

* NN require a lot of decisions:
  * How many layers?
  * How many hidden units?
  * What learning rate?
  * What activation functions?

* ML is a highly iterative process: start with idea and keep iterating.

* Usually take data and split into 3 sets:
  * Training
  * Hold-out, cross val, dev set.
  * Test set.

* Traditional ML: 60%/20%/20% split when dealing with 100, 1000 or 10000 items in the set..

* Neural nets tend to have much larger datasets (1m+), so split may be more like: 98%/%1/1%.
  * Want a dev set that let's you quickly iterate through algorithms to figure out quickly which ones are better.

## Bias / Variance

* high bias: underfitting.
* high variance: overfitting.
