---
title: ML Regression
date: 2016-02-03 00:00
category: reference/moocs
modified: 2023-04-09 00:00
---

Notes taken during [ML Regression](https://www.coursera.org/learn/ml-regression) by Coursera.

## Linear Algebra Refresher

### Matrices and Vectors

* Denoted by ``n x m`` (n rows, m columns).

### Special Matrices

* Square matrix: a matrix whose dimensions are `n x n`.
* Zero matrix: denoted by ``0n x m``, a matrix where all the values are 0. (acts like 0 in scalar land)
* Identity matrix: main diagonals are 1 and others 0. When multipling with another matrix, the result is the other matrix. (acts like 1 in scalar land)
* Column matrix: ``n x 1`` and row matrix: ``1 x n``. Aka vector.

### Arithmetic

* Adding or subtract matrices: add or subtract each field. Must be the same size. (see [Vector Addition](../../../permanent/vector-addition.md))
* Scalar multiplication: multiply each value in matrix by scalar (see [Vector Scaling](../../../permanent/vector-scaling.md)).
* Matrix multiplication: `A x B`
    * A must have same number of columns as B does rows eg ``A(2 x 3) * B(3 x 2)`` is valid. The resulting size will be ``2 x 2``.
  * Example:

        A = [1, 2, 4]   B = [4, 2]
          [4, 3, 5]       [5, 8]
                          [7, 1]

           (2 x 3)        (3 x 2)

        Outcome = [(1 * 4) + (2 * 5) + (4 * 7)]
                [(1 * 2) + (2 * 8) + (4 * 1)]
                [(4 * 4) + (3 * 5) + (5 * 8)]
                [(4 * 2) + (3 * 8) + (5 * 1)]

  * Commutative property does not apply in matrix multiplication: order matters.

### Determinant

* Function that takes a square matric and convert it to a number. Formula looks like:

```
[a c]
[b d] == a*d - c*b
```

### Matrix Inverse

* Given a square matrix `A` of size `n x n` we want to find another matrix such that:

`AB = BA = I n`

### Matrix calculus

* Each of the points in a matrix can be function, can determine the derivative of each point.

## Week 1 - Simple Regression

* Course Outline:
    * Module 1: Simple regression
        * Low number of inputs
        * Fit a simple line through the data
        * Need to define "goodness-of-fit" metric for each possible line.
        * Gradient descent algorithm:
            * Get estimated parameters
            * Interpret
            * Use to form predictions
      * Module 2: Multiple relationships
          * More complicated relationship than just a line.
          * Incorporate more inputs when training the model.
      * Module 3: Assessing performance
          * Determine when "overfitting" the data.
          * [Bias-Variance Tradeoff](../../../permanent/bias-variance-tradeoff.md)
              * Simple models are well behave but can be too simple to describe a behaviour accurately.
              * Complex models can have odd behaviour.
      * Module 4: Ridge Regression
          ``Ridge total cost = measure of fit + measure of model complexity``
          * Cross validation (??)
      * Module 5: Feature Selection & [Lasso](../../../permanent/lasso.md) Regression
          * What are the most important features for model?
           ``Lasso total cost = measure of fit + (different) measure of model complexity``
          * Will learn: coordinate descent algorithm.
      * Module 6: Nearest neighbor & kernel regression
          * Methods useful when you have a lot of data:
              * Nearest neighbor: find closest piece of data to what you're looking at and use for predictions
          * Kernel regression
* Assumed background
  * Basic calculus (LOL)
  * Basic linear algebra:
    * Vectors
    * Matrices
    * Matrix multiply

### Regression fundamentals

* Housing prices
    * Look at recent sales in the neighbour hood to help inform your house price.
* Data:
  * "For each house that sold recently, record square feet the house had and what the price was."
    * ``x1 = sq_ft, y2 = price````x2 = sq_ft, y2 = price``, ``x3 = sq_ft, y3 = price`` etc
  * Regression model: ``y[i] = f(x[i]) + E[i]`` - Outcome is the function applied to the input plus some error.
    * ``E(e[i]) = 0`` - expected value of error is 0 - will be equally likely to have positive or negative values.
* Tasks in regression:
  * Task 1: Determining options for data model.
  * Task 2: Estimate the specific fit from the data.
* Block diagram for regression (diagram from memory)

  ![Regression block diagram](/reference/_media/regression-block-diagram.jpg)

### ML Model

* Equation of a line: "intercept + slope * variable of interest" (``f(x) = W₀ + W₁ * Xᵢ``)
  * Single point includes the error (the distance from point back to line): (``f(x) = W₀ + W₁ * Xᵢ + εᵢ``).
  * Parameters of model = ``W₀`` (intercept) and ``W₁`` (slope)
    * Aka: regression coefficients.

### Quality metric

* "Cost" of using a given line.
  * Can be calculated with "Residual sum of squares (RSS)"
    * Take each error (``εᵢ``) - calculated as ``(actual_value - predicted value)²``
    * Add them for each point.
    * In pseudo:

      ```
      RSS(W₀, W₁) = sum([(y - (W₀ + W₁ * Xᵢ))² for y in actual_prices])
      ```

  * Goal is to search over all possible lines to find the lowest RSS.

### Using fitted line

* Estimated parameters:
  * ``Ŵ₀``
    * Intercept: value of ``y`` when ``x == 0``
      * In the context of square feet features, it'd be houses that are just land.
      * Usually not very meaningful (aka intercept is negative, does that mean land only sales mean the seller gets money??)
  * ``Ŵ₁``
    * Estimated slope
    * "Per unit change in input" - "when you go up 1 square foot, how much does the price go up?"

### Finding the best fit (minimizing the cost)

* ``min([RSS(W₀, W₁) for (W₀, W₁) in ALL_POSSIBLE_PARAMS])``
* Convex / concave functions:
  * Concave
    * Looks like arc of cave.
    * Line between 2 points lie below curve of function
    * Max is where derivative = 0
  * Convex
    * Opposite of concave
    * Any two points will be above the curve of the function
    * Min is where derivative = 0

      ![Concave Convex](/reference/_media/concave-convex.png)

  * Other functions:
    * Can be multiple solutions to derivative = 0 or no solutions
* Finding the max or min analytically:
  * Get derivate of function eg (``-2w + 20``).
  * Set derivate to 0 then solve for ``w``: (``-2w + 20 = 0 == w = 10``):
* Finding the max via hill climbing:
  * Idea:
    * Keep increasing ``w`` via some "step size" until "converged" eg: derivate = 0.
    * Divide space:
      * On one side, the derivative will be more than 0, on the other less.
    * In pseudo:

      ```
      # g = func
      # w = value

      def has_converged(g, w):
           return derivate(g, w) ~ 0

      while not has_converged(g, w):
          w = w + STEP_SIZE
      ```

    * While not converged:
      * Take previous w
      * Move in direction specified by derivated via some step size
  * Choosing stepsize and convergence criteria:
    * Fixed step size = set to constant size.
      * Can result in jumping over optimal, resulting in slow convergence.
    * Decrease the stepsize as you get closer to convergence.
    * Convergence will never get exactly to 0, but you set a threshold for convergence success.
      * Threshold depends on dataset.
    * Commons choices:
      * Algorithms that decrease with number of iterations:
        * ``η[t] = α / t`` (step size at position t is some alpha over t)
        * ``η[t] = α / sqrt(t)`` (step size at position t is some alpha over square root of t)

### Multiple dimensions: gradients

* When you have functions in higher dimensions, don't talk about "derivates" talk about "gradients".
* Notation example:

  ![Gradient Notation](/reference/_media/gradient-notation.png)

* Gradient definition:
  * A vector (array) with partial derivatives.
* Partial derivatives:
  * Same as derivate but involves other derivates. Treats them like constants.
* Worked example:

  ![Gradient Example](/reference/_media/gradient-example.png)

* Gradient descent == hill descent but compute the gradient instead of derivate for each iteration.

### Asymmetric cost functions

* "Prefer to under estimate than over"
  * Eg: don't want to predict my house price as too high.

## Week 2 - Multiple Regression

* Polynomial regression:
  * Even with only a single input variable, a linear equation model may not represent the relationship between the output variable.
  * Could use a higher-order function (quadratic, polynomial etc):
    * ``yᵢ = W₀ + W₁Xᵢ + W₂Xᵢ² ... + WpXᵢ^p + εi``
    * Treat different powers of ``x`` as "features":
      * feature 1 = ``1 (constant)``
      * feature 2 = ``x``
      * ..
      * feature p + 1 = ``x^p``
    * Different coefficients = parameters.
* More inputs added to regression model lets you factor in other relationships between your output variable.

![General notation](/reference/_media/general-notation.png)
![More notation](/reference/_media/more-notation.png)

  * |Question| What letter will be used to represent number of observations?
    * ``N``
  * |Question| What letter will be used to represent no of inputs? (``x[i]``)
    * ``d``
  * |Question| What letter will be used to represent no of features? (``hⱼ(x)``)
     * ``D``
* Commonly used equation:

  ![Commonly used equation](/reference/_media/commonly-used-equation.png)

* Interpreting coefficients of fitted function
  * Co-efficient's should be considered in the context of the entire model.
    * Eg: Number of bedrooms might have a negative coefficient if the square feet of the house is low.
  * If in a situation where you can't "fix" an input (eg if all features are a power of one input), then you can't interpret the coefficient.
* Linear Algebra Review
* Stages for computing the least squares fit:
  * Step 1: Rewrite regression model using linear algebra
    * Rewrite the regression model for a single observation in matrix notation:

      ![Rewrite Matrix Notation](/reference/_media/rewrite-matrix-notation.png)

      * Multiple two vectors:

        1. A row vector (aka transposed column vector) of parameters / coefficients
        2. A column vector of features.

      * Then add the error term.
    * Rewrite model for all observations:

      ![Matrix Notation All Observations](/reference/_media/matrix-notation-all-observations.png)

        * Rows of middle matrix (matrix ``H``) = vector of features from previous section.
  * Step 2: Compute the cost
    * Algorithm = search all different fits to find the smallest cost (RSS).
    * RSS in matrix notation:

      ![RSS Matrix Notation](/reference/_media/rss-matrix-notation.png)

      * ``y - Hw`` is a residual vector.
      * Using vector multiplication of that vector times the transpose, you end up with a scalar of the residual sum of squares.
  * Step 3: Take the gradient of the RSS:
    * ``-2 * H_vector_transposed * (y_vector - H_vector * w_vector)``

      ![Gradient of RSS Notes](/reference/_media/gradient-of-RSS-notes.png)

  * Step 4: Approach 1, closed-form solution: solve for W.
    * Set result to 0 and solve.
    * Matrix inverse * matrix = identity matrix (??)

## Week 3 - Assessing Performance

* Goal: figure out how much you are "losing" using your model compare to perfection.
	* Example: low predictions causing house to be listed too cheap.
* Loss can be measured with a *loss function*: $$L(y, f_\hat{w}(\mathbf{x})) $$
    * Examples:
        * Absolute error: $$L(y, f_\hat{w}(\mathbf{x})) = |y - f_\hat{w}(\mathbf{x})| $$
            * [Mean Absolute Error](../../../permanent/mean-absolute-error.md)
        * Squared error: $$L(y, f_\hat{w}(\mathbf{x})) = (y - f_\hat{w}(\mathbf{x}))^2 $$
            * [Root Mean-Squared Error](../../../permanent/root-mean-squared-error.md) squared.
                * Can have a very high cost if difference is large, compared to absolute error.
* Compute training error:
  1. Define some loss function (as above).
  2. Computing training error.
        * Example: Average loss on training set using squared error:
            * = $$1/N \sum\limits_{i=1}^{N} L(y, f_\hat{w}(\mathbf{x}))$$
* average of loss function

            * = $$1/N \sum\limits_{i=1}^{N} (y - f_\hat{w}(\mathbf{x}))^2$$

* average of squared error

            * = $$\sqrt{1/N \sum\limits_{i=1}^{N} (y - f _\hat{w}(\mathbf{x}))^2}$$

(convert to root mean squared error (RMSE))
            * provides more intuitive format (dollars, instead of squared dollars)*
* Training error vs model complexity:
    * Training error obviously is lowers as complexity of model increases (can get almost perfect fit with high complexity models):
        * Doesn't necessarily mean that predictions will be good, in fact they can get extremely bad with overfit models.
            * Summary: low training error != good predictions.
* * Generalisation error*: theoretic idea for figuring out ideal model.
    * Weight house price pairs ``(house, price)`` by how they are to occur in dataset and use to evaluate predictions, use to evaluate predictions.
		* For a given square footage, how likely is it to occur in the dataset?
		* For houses with a given square footage, what house prices are we likely to see?
	* Theoretical idea: impossible to compute *generation error*: requires every possible dataset in existence.
* *Test error*: like generalisation error but actually computable.
	* Basically, use *test error* to roughly approximate generation error.
	* Average loss on houses in test set: $$1/Ntest \sum_\limits{i=1}^{Ntest} L(y, f_\hat{w}(\mathbf{x})) $$
		* Note: $$f_\hat{w}(\mathbf{x}) $$

 was fit with training data.

* Defining overfitting:
	* Overfitting if there exists a model with estimated params $$w'$$

such that:

		1. Training error ($$\hat{w}$$

) < Training error ($$w'$$

).

		2. True error ($$\hat{w}$$

) > True error ($$w'$$

)
	* In other words: overfit if training error is low compared to another model but "true error" is high. Better to have high training error but low "true error".
* Training/test split: how to think about dividing data between training and test split.
	* General rule of thumb: just enough points in test set to approximate generalisation error well.
		* *To do: figure out how to approximate generalisation error.*
		* If you don't have enough for training data, then need to use other methods like "cross validation".
* 3 sources of error:
	1. Noise
	2. Bias
	3. Variance

* Noise:
    * Data is inherently noisy; there are heaps of variables you can't factor into model.
        * Variance of noise term: spread of $$\epsilon_i$$
    * Can't really control noise, there will be factors that influence observations outside of feature set.
* Bias:
	* Difference between fit and "true function".
		* "Is our model flexible enough to capture true relationship between data and model."
	* $$Bias(x) = f_{w(true)}(\mathbf{x}) - f_\hat{w}(\mathbf{x}) $$
	* Bias is basically how good our fit is; more parameters generally means less bias.
* Variance:
	* How much variance does model have?
	* High complexity models == high variance.
	* Variance is basically how wild our model is; more parameters means higher variance.
* Bias-variance tradeoff:
	* As error decreases, bias decreases and variance increases.
	* $$MSE = bias^2 + variance $$

(mean-squared error)
	* Want to find sweet spot in model where bias and variance are together as low as possible.
	* We can't compute "true function" but there are ways to optimise an approximation a practical way.
* Error vs amount of data:
	* True error decreases as data points in training set increase to limit of bias + noise.
	* Training error goes up as data points increase to same limit.
* Validation set:
	* Choosing tuning parameters, $$\lambda $$

(eg degree of polynomial):

		* Naive approach: For each model complexity $$\lambda $$

:
			1. Estimate params on training data.
			2. Assess performance on test data.

			3. Choose $$\lambda $$

with lowest test

			* Problem with this the test data was already used for selecting $$ \lambda $$

* will be overfit.
		* Better approach: split data 3 ways: training, validation and test set

			* Select $$\lambda $$
that minimizes error on **validation set**.

			* Get approximate generalisation error of $$ \hat{w}_{\lambda\star} $$
 using **test set**.
			* Typical splits:
				* 80% / 10% / 10%
				* 50% / 25% / 25%

## Week 4 - Ridge Regression

i* Symptom of overfitting:
	* When model is extremely overfit, magnitude of coefficients can be extremely large.
* Overfitting not unique to polynomial regression: can also occur if you have lots of inputs.
* Number of observations can influence overfitting:
	* Few observations (N small) can cause model to be quickly overfit as complexity grows.
	* Many observations (N very large) can be harder to overfit (but harder to find datasets).

        ![Number of Observations](./images/number-of-observations.png)

* The larger the inputs, the more change of data not including inputs for all data points causing overfitting.
* Balancing fits and magnitude of coefficients:
	* Can improve quality metric by factoring in coefficient magnitude to avoid complex models.
	* ``Total cost = measure of fit (RSS) + measure of coefficient magnitudes``.
	* Want to find the ideal balance between the two.
	* Ways to measure coefficient magnitude:
		* Add coefficients: $$\hat{w}_0 + \hat{w}_1 ... \hat{w}_d $$
			* the negative coefficients would cancel out the others.
		* Add the abs value of coefficients: $$\sum\limits_{i=0}^{D} | \hat{w}_i | $$
			* Aka "L1 norm"
		* Sum squared values of coefficients: $$\sum\limits_{i=0}^{D} (\hat{w}_i)^2 $$

(sum of squared values of coefficients)
			* Aka "L2 norm"
* Resulting ridge objective and its extreme solution.
	* Can use an extra parameter to tune how much weight is applied to the measure of coefficient magnitude: $$RSS(\mathbf{w}) + \lambda ||\mathbf{w}||_2^2 $$
	* Set the param to 0 = only RSS would weight in quality metric.
	* Set the param to $$\infty $$, $$\mathbf{w} = 0 $$

would be the solution.

	* Need to find a balance between the two: enter "ridge regression" (aka $$L_2 $$

regularisation.

* How ridge regression balances bias and variance.
	* High $$\lambda $$

= high bias, low variance.

		* As $$\lambda $$

get closer to infinitely, coefficients get smaller and less general.

	* Low $$\lambda $$

= low bias, high variance.

		* As $$\lambda $$

get closer to 0, coefficients get larger (RSS takes over) and shit gets crazy.

* Ridge regression demo
	* "Leave one out" (LOO) cross validation can show approximate average mean squared error (MSE) and is a useful technique for choosing $$\lambda $$

.

* The ridge coefficient path:
  * Graphical representation of how the value of lambda affects the coefficient:

    <img src="./images/coefficient-path.png"></png>

    Again, at 0 it's just the RSS ($$\hat{w}_{LS} $$

), as it gets larger, they approach 0.

  * Some sweet spot between smallish coefficients and a well fit model.
* Computing the gradient of the ridge objective:
  * Can rewrite the L2 norm ($$|| \mathbf{w} ||_2^2 $$) in vector notation as follows: $$w^tw $$

.

  * Then, the entire ridge regression total cost can be rewritten like: $$y - (\mathbf{H}\mathbf{w})^2(\mathbf{H}\mathbf{w}) + || \mathbf{w} ||_2^2 $$

, which is just the RSS + the L2 norm.

  * Can take the gradient of both terms and you get: $$-2\mathbf{H}^T( y - (\mathbf{H}\mathbf{w})) + 2\mathbf{w} $$

.

  * The gradient of the L2 norm is analygous to the 1d case: derivate of $$w^2 $$

= $$2w $$

* Approach 1: closed-form solution:
	* Summary: set the gradient = 0 and solve for $$\hat{w} $$

.
	* Steps:

		1. Multiple the $$\mathbf{w} $$

vector by the identity matrix to make the derivation easier.

		  $$\triangle cost(\mathbf{w})) = -2\mathbf{H}^T(\mathbf{y} - \mathbf{H}\mathbf{w}) + 2\lambda\mathbf{I}\mathbf{w} = 0 $$

		2. Divide both sides by 2.

			$$ =-\mathbf{H}^T(\mathbf{y} - \mathbf{H}\mathbf{w}) + \lambda\mathbf{I}\mathbf{w} = 0 $$

		3. Multiple out.

			$$-\mathbf{H}^T\mathbf{y} + \mathbf{H}^T \mathbf{H}\mathbf{w} + \lambda\mathbf{I}\mathbf{w} = 0 $$

		4. Add $$\mathbf{H}^T\mathbf{y} $$

to both sides.

			$$\mathbf{H}^T \mathbf{H}\mathbf{w} + \lambda\mathbf{I}\mathbf{w} = \mathbf{H}^T\mathbf{y}  $$

		5. Since the $$ \hat{w} $$

appears in both expressions, can factor it out.

			$$(\mathbf{H}^T \mathbf{H} + \lambda\mathbf{I})\mathbf{w} = \mathbf{H}^T\mathbf{y} $$

		6. Multiple both sides by the inverse of $$\mathbf{H}^T \mathbf{H} + \lambda\mathbf{I} $$

.

			$$\mathbf{w} = (\mathbf{H}^T \mathbf{H} + \lambda\mathbf{I})^{-1}\mathbf{H}^T\mathbf{y} $$

* Discussing the closed-form solution

        * Can prove the closed-form solution is congruent with the above notes, by setting $$\lambda = 0 $$
and seeing that results are equal to the least squares closed form solution:

	 	$$\hat{w}^{ridge} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{y} = \hat{w}^{LS} $$-results 

	* Setting lambda to infinity results in 0 because the inverse of infinity matrix is like dividing by infinity.

	* Recall previous solution for $$\hat{w}^{LS} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{y} $$
:

		* Invertible if number of linear independant observations is more than number of features ($$ N > D $$
) - an added bonus of using ridge regression.

		* Complexity of the inverse: $$O(D^3) $$

	* Properties for ridge regression solution:

		* Inverible always if $$\lambda > 0 $$even if $$N < D $$
.
		* Complexity of inverse is the same. May be prohibitive to do this solution with lots of features.

		* Called "regularised" because adding the $$\lambda\mathbf{I} $$is "regularising" the $$\mathbf{H}^T \mathbf{H} $$
, making it invertable ever with lots of features.

* Approach 2: gradient descent
	* Same as RSS gradient descent but factors in the L2 norm cost component gradient.
* Selecting tuning parameters via cross validation.
	* If you don't have enough data to build a validation set, for testing the performance of $$\lambda $$

, you can grab the validation set as a subset of the training data, but instead of grabbing a single slice, can grab all slices and average the results.

* K-fold cross validation.
	* Algorithm:
		* For each lambda you're considering:
			* For each data slice of size $$N/K $$

:
				* Go through each validation block on training data compute predictions.

				* Compute error for predictions using the $$\lambda $$

value.

			* Calculate average error: $$CV(\lambda) = 1/K \sum\limits_{k=1}^K error_K(\lambda) $$

	* Choosing ideal value of K:
		* Best approximation occurs for validation sets of size 1 (K = N) aka "leave-one-out" (LOO) cross validation.

		* Computationally intensive: requires computing N fits of model per $$\lambda $$

.

		* Common to use $$K=5 $$

(5-fold CV) or $$K=10 $$

(10-fold CV) if $$K=N $$

is computational infeasible to run.

* How to handle intercept term.
	* The goal of the ridge regression is to lower the coefficients values $ but that doesn't necessarily make sense for $$\hat{w}_0 $$

.
	* Solutions:

		* Option 1: leave out the $$\hat{w}_0 $$

coefficient when computing the optimal fit.
			* In closed-form solution, can use a modified identity matrix with the first position set to 0.

			* In gradient descent, can just skip the ridge component when computing the j value for $$\hat{w}_0 $$

.
		* Option 2: transform data so the expected intercept value is around 0 (called "centring").

## Week 5 - Feature Selection & Lasso

* Feature selection task motivation
	* Efficiency
		* when faced with large feature sets, predictions can get expensive.
		* When $$\hat{w} $$

is sparse (eg has a lot of 0s in the dataset) can eliminate all 0 features.
	* Interpretability
		* which features are relevant to housing predictions?
* All subsets algorithm
	* Search over every combination of features computing the RSS:
		* Iterate through every N = 1 sized combination, then N = 2 and so on up to feature D.
	* Slow to compute with a lot of features: complexity = $$2^{D} $$
		* With 15 features $$2^{15} = 32768 $$
* Greedy algorithms
	* "Forward stepwise algorithm"
		* Find best N = 1 feature subset (maybe based on training error)
		* Next, find best N = 2 subsets that includes previous feature.
	* Use *validation set* or *cross validation* to choose when to stop greedy procedure - training error alone is insufficient.
	* Quicker to compute than all subsets, but may not find optimal combination.
		* Complexity:
			* 1st step: D models, 2nd step: D - 1 models, 3rd step: D - 2 models etc
			* Worst case complexity: $$D^2 $$
	* Other greedy algorithms
		* Backward stepwise: start with full model and work backwards removing one.
		* Combining forward and backward steps: add steps to remove features not considered relevant.
* Regularised regression for feature selection
	* Ridge regression (L2 penalty) encourages small weights but not exactly 0.
	* Core idea: can we use a similar idea but actually get weights to 0 and effectively remove them from the model?
	* Potential method 1: Thresholding.
		* Remove features that falls below some threshold for $$w $$

.
		* Problem: redundant features
			* If you have bathroom and number of showers, the weights would be distributed over both features. If you remove one, the coefficient should double on the other.
			* This means you could potentially end up discarding all redundant features.
	* Potential method 2: use L1 norm for regularized regression aka "Lasso regression" aka "L1 regularised regression".

		* $$ RSS(\mathbf{w}) + \lambda * ||\mathbf{w}||_1 $$

			* When tuning param, $$\lambda $$is 0, $$\hat{w}^{lasso} = \hat{w}^{LS} $$

 (ie result is just least squares.

			* When $$\lambda = \infty $$

, $$\hat{w}^{lasso} = 0 $$

 (ie coefficients shrink to 0).
		* Coefficient path for Lasso:

			![Coefficient path for Lasso](coefficient-path-for-lasso.png)

			* For certain lambda values, features jump out of the model and other fall to 0. Individual impact of features becomes clearer.
* Optimising the lasso objective
	* The derivate of $$|w_j| $$can't be calculated when $$|w_j| = 0 $$

. Therefore, can't calculate the derivate.
	* Use "subgradients" over gradients.
* Coordinate descent
	* Goal: minimise a function $$g(w_0, w_1, ..., W_D) $$

, a function with multivariables.
	* Hard to find minimum for all coordinates, be easy for a single coordinate with all the other fixed..
	* Algorithm:
		* While not converged:
			* Pick a coordinate (w coefficient) and fix the other.
			* Find the minimum of that function.
	* No stepsize required
	* Converges for lasso objective
	* Converges to optimum in some cases.
	* Can pick next coordinate at random, round robin or something smarter...
* Normalising features
	* Idea: put all features into same numeric range (eg number of bathrooms, house square feet)
	* Apply to a *column* of the feature matrix.
	* Need to apply to training and test data.
	* Formula:

		$$\underline{h}_j(\mathbf{x}_k) = \frac{h_j(\mathbf{x}_k)}{ \sqrt{\sum\limits_{i=1}^{N} h_j(\mathbf{x}_i)^2}} $$

* Coordinate descent for least squares regression
	* For each j feature, fix all other coordinates $$ \mathbf{w}_{-j} $$and take the partial with respect to $$\mathbf{w}_j $$
	* Gradient derived to $$-2P_j + 2w_j $$($$P_j $$

== residual without jth feature)

	* Setting to 0 and solving for $$\hat{w}_j $$

results in $$\hat{w}_j = P_j $$

	* Intuition: if the residual between the predictions without j and the actual prediction is large, then the weight of j will be large and vice versa (to do: check this as your understanding improves).
* Coordinate descent for lasso (for normalised features)
	* Same as cord descent for least squares, except we set $$\hat{w}_j $$

using a "thresholding" function (in code):

		```
		def set_w_j(p_j, lambda):
          if p_j < -lambda / 2:
              return p_j + lambda / 2
          if -lambda / 2 < p_j < lambda / 2:
              return 0
          if p_j > lambda / 2:
              return p_j - lambda / 2
		``` 

	* General idea here is "soft thresholding", aiming to get values to 0 that fit within some range:

		![Soft thresholding]("./images/cord-descent-soft-thresholding.png")

* Coordinate descent for lasso (for unnormalised features)
	* Normalisation factor is used during the set $$w_j $$

portion of the regression.

* Choosing the penalty strength and other practical issues with lasso.
	* Same as ridge regression:
		* If enough data, validation set.
		* Compute average error
* Summary
	* Searching for best features
		* All subsets
		* Greedy algorithms
		* Lasso regularised regression approach
	* Contrast greedy and optimal algorithms
	* Describe geometrically why L1 penalty leads to sparsity
	* Estimate lasso regression parameters using an iterative coordinate descent algorithm
	* Implement K-fold cross validation to select lasso tuning parameter $$\lambda $$
* Note: be careful about interpreting features, need to consider in the context of the entire model.

## Week 6 - Nearest Neighbors & Kernel Regression

* Limitations of parametric regression
  * A polynomial fit might work well in certain regions of input space and not well in others.
    * Eg cubic fit might work well for higher square feet houses, but not so well for lower.
  * Ideal method:
    * Flexible enough to support "local structure" ie different fit at certain input space regions.
    * Doesn't require us to infer "structure breaks" ie the places where we want a different fit.
* Nearest neighbour regression approach
  * For some input, find the closest observation. That's it.
  * What people do naturally when predicting house prices.
  * Formally:

	1. For some input, find closest $$\mathbf{x}_i $$

using some distance metric.

	2. Predict: $$\hat{y}_q = y_{nn} $$

* Distance metrics
	* 1D feature sets: Euclidian distance
		* $$distance(x_j, x_q) = |x_j - x_q| $$
	* Multi dimensions:
		* Can use a weight on each feature to determine importance in computing distance ie sqft closeness is important, but year renovated may not be.
		* $$ distance(\mathbf{x_j}, \mathbf{x_q}) = \sqrt{a_1(x_i[1] - x_q[1])^2) + .. a_D(x_i[D] - x_q[D])^2)} $$
			* Note: $$a_D $$

is the "weight" in this equation.

* 1-Nearest Neighbour Algorithm
	* Need to define your distance function.
	* In pseudo:

		```
		closest_distance = float('inf')

		for i in all_data_points:
			dist = calculate_distance(i, input)
           if dist < closest_distance:
                closest_distance = dist

		return closest_distance
 			
		``` 

	* Notes:
		* Sensitive to regions with missing data; need heaps of data for it to be good.
		* Sensitive to noisy datasets.
* k-Nearest neighbours regression
	* Same as 1-Nearest Neighbour but average over values of "k"-nearest neighbours.
* k-Nearest neighbours in practise
	* Usually has a much better fit than 1-nearest.
	* Issues in regions where there's sparse data and at boundaries.
* Weighted k-nearest neighbours
	* Idea: weight neighbours by how close or far they are to data point.
	* Formula (where $$C_{qNN1} $$

refers to some weight for NN1:

		$$\hat{y}_q = \dfrac{C_{qNN1}Y_{qNN1} + C_{qNN2}Y_{qNN2} .. C_{qNNK}Y_{qNNK}}{\sum\limits_{j=1}^{K}C_{qNNj}} $$

	* Weighting data points:

		* Could just use inverse of distance: $$C_{qNN1} = \dfrac{1}{distance(\mathbf{X_j}, \mathbf{X_q})} $$

		* Can use "kernel" functions:
			* Gaussian kernel
			* Uniform kernel etc
* Kernel regression
	* Instead of weighing n-neighbours, weigh all points with some "kernel":

		$$\hat{y}_q \frac{\sum\limits_{y=1}^{N} C_{qi}Y_{qi}}{\sum\limits_{y=1}^{N} C_{qi}} = \frac{\sum\limits_{y=1}^{N} kernel_{\lambda}(distance(\mathbf{x_i},\mathbf{x_q})) * y_i}{kernel_{\lambda}(distance(\mathbf{x_i},\mathbf{x_q}))} $$

	* In stats called "Nadaya-Watson" kernel weighted averages.
	* Need to choice a good "bandwidth" (lambda value)
		* Too high = over smoothing; low variance, high bias.
		* Too low = over fitting; high variance, high bias.
	* Need to choose kernel but bandwidth more important.
	* Use validation set (if enough data) or cross-validation to choose $$\lambda $$

value.

* Global fits of parametric models vs local fits of kernel regression
  * If you were to predict datapoint by averaging all observations, you'd end up with a constant fit.
  * Kernel gives constant fit at a single point; a "locally constant fit".
