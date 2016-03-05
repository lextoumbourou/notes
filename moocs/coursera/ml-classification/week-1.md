# Week 1

## Assumed background

* Course 1 & 2 in ML Specialisation.
* Maths
  * Basic calculus
  * Basic vectors
  * Basic functions
    * Exponentiation e^x
    * Logarithms

## Linear classifiers

### Motivating example

* Capturing sentiment of restaurant reviews

### Intuition behind linear classifiers

* Use training data to figure out coefficients of words like:

  ```
  good     |    1.0
  great    |    1.5
  bad      |   -1.0
  terrible |   -2.0
  ```

  then use that data to predict sentiment on other texts.

* Also should use validation set for verifying models.

### Decision boundaries

* Suppose you have a model with two coefficients:

  ```
  good  |  1.0
  bad   | -2
  ```

  The line between a positive and negative sentiment would sit at: ``1 * #good - 2 * #bad``. Anything above the line is positive and below is negative.

* Decision boundary types:
  * When 2 coefficients are non-zero: line
  * When 3 coefficients are non-zero: plane
  * When many coefficients are non-zero: hyperplane
  * More general classifiers: more complicated (squiggly?) lines

### Linear classifier model
