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

### Effect of coefficient values on decision boundaries

* Increasing intercept value, shifts decision boundary up.
* Increase value of coefficients can change the d.b. curve.

### Using features

* Features can modify impact of words in a sentence.
* With feature added, equation looks like:

$$ Score(\mathbf{x_i}) = w_0 h_0(\mathbf{x_i}) + .. + W_d h_D(\mathbf{x_i})) = \mathbf{w}^Th(\mathbf{x}_i) $$

* Flow chart summary:
	1. Get some training data.
	2. Extract features (word counts, td-idf etc).
	3. Generate a model with said features (create coefficients).
	4. Validate model (test data etc).
	5. Make predicts using model. 

### Class probabilities

* So far prediction = +1 or -1
* How do you capture confidence that something is definitely positive or maybe negative etc? Enter probability.

### Basics of probability

* Probability a review is positive is 0.7	
    * Interpretation: 70% of rows have y = +1
* Can interpret probabilities as "degrees of belief" or "degrees of sureness".
* Fundamental properties:
	* Probabilities should always be between 0 and 1 (no negative probabilities).
	* A set of probabilities classes should always add up to 1:

		$$ P(y=+1) + P(y=-1) = 1 $$

###	Basics of conditional probability

* Probability y is one given input can be represented as:

  $$ P(y=+1|\mathbf{x}_i) $$ where $$ \mathbf{x}_i $$ is some sentence.

* Conditional probability should always be between 0 & 1.
* Classes of conditional probabilities sum up to 1 over y:

  $$ P(y=+1|\mathbf{x}_i) + P(y=-1|\mathbf{x}_i) = 1 $$
  
  However, they don't add up to 1 for all $$ \mathbf{x} $$:

  	$$ \sum\limits_{X}^{} P(y=+1|\mathbf{x}_i) \neq 1 $$

### Using probabilities in classification

* A lot of classifiers output degree of confidence.
* We generally train a classifier to output some $$ \mathbf{\hat{P}} $$ which uses $$ \mathbf{\hat{w}} $$ values to make predictions. It can then use outputted probability to return +1 if > 0.5 (is positive) or -1 if < 0.5 (is negative) and also how confident we are with the answer.