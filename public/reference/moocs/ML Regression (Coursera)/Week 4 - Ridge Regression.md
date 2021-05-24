* Symptom of overfitting:
	* When model is extremely overfit, magnitude of coefficients can be extremely large.

* Overfitting not unique to polynomial regression: can also occur if you have lots of inputs.

* Number of observations can influence overfitting:
	* Few observations (N small) can cause model to be quickly overfit as complexity grows.
	* Many observations (N very large) can be harder to overfit (but harder to find datasets).

        ![number of observations](./images/number-of-observations.png)

* The larger the inputs, the more change of data not including inputs for all data points causing overfitting.

* Balancing fits and magnitude of coefficients:
	* Can improve quality metric by factoring in coefficient magnitude to avoid complex models.
	* ``Total cost = measure of fit (RSS) + measure of coefficient magnitudes``.
	* Want to find the ideal balance between the two.
	* Ways to measure coefficient magnitude:
		* Add coefficients: $$ \hat{w}_0 + \hat{w}_1 ... \hat{w}_d $$
			* the negative coefficients would cancel out the others.
		* Add the abs value of coefficients: $$ \sum\limits_{i=0}^{D} | \hat{w}_i | $$
			* Aka "L1 norm"
		* Sum squared values of coefficients: $$ \sum\limits_{i=0}^{D} (\hat{w}_i)^2 $$ (sum of squared values of coefficients)
			* Aka "L2 norm"

* Resulting ridge objective and its extreme solution.
	* Can use an extra parameter to tune how much weight is applied to the measure of coefficient magnitude: $$ RSS(\mathbf{w}) + \lambda ||\mathbf{w}||_2^2 $$
	* Set the param to 0 = only RSS would weight in quality metric.
	* Set the param to $$ \infty $$, $$\mathbf{w} = 0 $$ would be the solution.
	* Need to find a balance between the two: enter "ridge regression" (aka $$ L_2 $$ regularisation.

* How ridge regression balances bias and variance.
	* High $$ \lambda $$ = high bias, low variance.
		* As $$ \lambda $$ get closer to infinitely, coefficients get smaller and less general.
	* Low  $$ \lambda $$ = low bias, high variance.
		* As $$ \lambda $$ get closer to 0, coefficients get larger (RSS takes over) and shit gets crazy.

* Ridge regression demo
	* "Leave one out" (LOO) cross validation can show approximate average mean squared error (MSE) and is a useful technique for choosing $$ \lambda $$.

* The ridge coefficient path:
  * Graphical representation of how the value of lambda affects the coefficient:

    <img src="./images/coefficient-path.png"></png>

    Again, at 0 it's just the RSS ($$ \hat{w}_{LS} $$), as it gets larger, they approach 0.

  * Some sweet spot between smallish coefficients and a well fit model.

* Computing the gradient of the ridge objective:
  * Can rewrite the L2 norm ($$ || \mathbf{w} ||_2^2 $$) in vector notation as follows: $$ w^tw $$.
  * Then, the entire ridge regression total cost can be rewritten like: $$ y - (\mathbf{H}\mathbf{w})^2(\mathbf{H}\mathbf{w}) + || \mathbf{w} ||_2^2 $$, which is just the RSS + the L2 norm.
  * Can take the gradient of both terms and you get: $$ -2\mathbf{H}^T( y - (\mathbf{H}\mathbf{w})) + 2\mathbf{w} $$ .
  * The gradient of the L2 norm is analygous to the 1d case: derivate of $$ w^2 $$ = $$ 2w $$

* Approach 1: closed-form solution:

	* Summary: set the gradient = 0 and solve for $$ \hat{w} $$.
	* Steps:
		1. Multiple the $$ \mathbf{w} $$ vector by the identity matrix to make the derivation easier.

		  $$ \triangle cost(\mathbf{w})) = -2\mathbf{H}^T(\mathbf{y} - \mathbf{H}\mathbf{w}) + 2\lambda\mathbf{I}\mathbf{w} = 0 $$

		2. Divide both sides by 2.

			$$  =-\mathbf{H}^T(\mathbf{y} - \mathbf{H}\mathbf{w}) + \lambda\mathbf{I}\mathbf{w} = 0 $$

		3. Multiple out.

			$$ -\mathbf{H}^T\mathbf{y} + \mathbf{H}^T \mathbf{H}\mathbf{w} + \lambda\mathbf{I}\mathbf{w} = 0 $$

		4. Add $$ \mathbf{H}^T\mathbf{y} $$ to both sides.

			$$ \mathbf{H}^T \mathbf{H}\mathbf{w} + \lambda\mathbf{I}\mathbf{w} = \mathbf{H}^T\mathbf{y}  $$

		5. Since the $$  \hat{w} $$ appears in both expressions, can factor it out.

			$$ (\mathbf{H}^T \mathbf{H} + \lambda\mathbf{I})\mathbf{w} = \mathbf{H}^T\mathbf{y} $$

		6. Multiple both sides by the inverse of $$ \mathbf{H}^T \mathbf{H} + \lambda\mathbf{I} $$.

			$$ \mathbf{w} = (\mathbf{H}^T \mathbf{H} + \lambda\mathbf{I})^{-1}\mathbf{H}^T\mathbf{y} $$

* Discussing the closed-form solution
        * Can prove the closed-form solution is congruent with the above notes, by setting $$ \lambda = 0 $$ and seeing that results are equal to the least squares closed form solution:

	 	$$ \hat{w}^{ridge} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{y} = \hat{w}^{LS} $$ -results 

	* Setting lambda to infinity results in 0 because the inverse of infinity matrix is like dividing by infinity.
	* Recall previous solution for $$ \hat{w}^{LS} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{y} $$ :
		* Invertible if number of linear independant observations is more than number of features ($$  N > D $$) - an added bonus of using ridge regression.
		* Complexity of the inverse: $$ O(D^3) $$
	* Properties for ridge regression solution:
		* Inverible always if $$ \lambda > 0 $$ even if $$ N < D $$.
		* Complexity of inverse is the same. May be prohibitive to do this solution with lots of features.
		* Called "regularised" because adding the $$ \lambda\mathbf{I} $$ is "regularising" the $$ \mathbf{H}^T \mathbf{H} $$, making it invertable ever with lots of features.

* Approach 2: gradient descent
	* Same as RSS gradient descent but factors in the L2 norm cost component gradient.

* Selecting tuning parameters via cross validation.
	* If you don't have enough data to build a validation set, for testing the performance of $$ \lambda $$, you can grab the validation set as a subset of the training data, but instead of grabbing a single slice, can grab all slices and average the results.

* K-fold cross validation.
	* Algorithm:
		* For each lambda you're considering:
			* For each data slice of size $$ N/K $$:
				* Go through each validation block on training data compute predictions.
				* Compute error for predictions using the $$ \lambda $$ value.
			* Calculate average error: $$ CV(\lambda) = 1/K \sum\limits_{k=1}^K error_K(\lambda) $$
	* Choosing ideal value of K:
		* Best approximation occurs for validation sets of size 1 (K = N) aka "leave-one-out" (LOO) cross validation.
		* Computationally intensive: requires computing N fits of model per $$ \lambda $$.
		* Common to use $$ K=5 $$ (5-fold CV) or $$ K=10 $$ (10-fold CV) if $$ K=N $$ is computational infeasible to run.

* How to handle intercept term.
	* The goal of the ridge regression is to lower the coefficients values $ but that doesn't necessarily make sense for $$ \hat{w}_0 $$.
	* Solutions:
		* Option 1: leave out the $$ \hat{w}_0 $$ coefficient when computing the optimal fit.
			* In closed-form solution, can use a modified identity matrix with the first position set to 0.
			* In gradient descent, can just skip the ridge component when computing the j value for $$ \hat{w}_0 $$.
		* Option 2: transform data so the expected intercept value is around 0 (called "centring").
