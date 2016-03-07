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

* One you have your coefficients for each word, multiply each by the count of words and add together to get the score.

  ```
  Score(x) =
    word_count = get_word_count(sentence)
    W = get_coefficients_for_model()
    Score = W[0] + W[1] * word_count["awesome"] + W[2] * word_count["awful"]  # ...
  ```

* Wrap ``Score()`` in a ``sign()`` function for producing a binary result (1 if above 0 or -1 if below).

***

* Course notation is the same as used in Regression course.
* Can rewrite expression using course notation like:


$$ Score(\mathbf{x_i}) = w_0 + w_1 \mathbf{x_i}[1]] + .. + W_d \mathbf{x_i}[d] = \mathbf{w}^T\mathbf{x}_i $$

## Effect of coefficient values on decision boundaries

* Increasing intercept value, shifts decision boundary up.
* Increase value of coefficients can change the d.b. curve.

## Using features

* Features can modify impact of words in a sentence.
* With feature added, equation looks like:

$$ Score(\mathbf{x_i}) = w_0 h_0(\mathbf{x_i}) + .. + W_d h_D(\mathbf{x_i})) = \mathbf{w}^Th(\mathbf{x}_i) $$

* Flow chart summary:
	1. Get some training data.
	2. Extract features (word counts, td-idf etc).
	3. Generate a model with said features (create coefficients).
	4. Validate model (test data etc).
	5. Make predicts using model. 
