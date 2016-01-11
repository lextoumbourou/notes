## Week 3

* Goal: figure out how much you are "losing" using your model compare to perfection.

	* Example: low predictions causing house to be listed too cheap.

* Loss can be measured with a *loss function*: $$ L(y, f _\hat{w}(\mathbf{x})) $$

	* Examples:

	   * Absolute error:  $$ L(y, f _\hat{w}(\mathbf{x})) = |y - f _\hat{w}(\mathbf{x})| $$
	   * Squared error: $$ L(y, f _\hat{w}(\mathbf{x})) = (y - f _\hat{w}(\mathbf{x}))^2 $$
		   * Can have a very high cost if difference is large, compared to absolute error.

* Compute training error:

  1. Define some loss function (as above).
  2. Computing training error.
  
	  * Example: Average loss on training set using squared error: 
	      * = $$1/N \sum\limits_{i=1}^{N} L(y, f _\hat{w}(\mathbf{x}))$$ - average of loss function
	      * = $$1/N \sum\limits_{i=1}^{N} (y - f _\hat{w}(\mathbf{x}))^2$$ - average of squared error
	      * = $$\sqrt{1/N \sum\limits_{i=1}^{N} (y - f _\hat{w}(\mathbf{x}))^2}$$ (convert to root mean squared error (RMSE))
		     *provides more intuitive format (dollars, instead of squared dollars)*

* Training error vs model complexity:

	* Training error obviously is lowers as complexity of model increases (can get almost perfect fit with high complexity models):
		* Doesn't necessarily mean that predictions will be good, in fact they can get extremely bad with overfit models.
	   * Summary:  low training error != good predictions.

* *Generalisation error*: theoretic idea for figuring out ideal model.

	* Weight house price pairs ``(house, price)`` by how they are to occur in dataset and use to evaluate predictions, use to evaluate predictions.
		*  For a given square footage, how likely is it to occur in the dataset?
		* For houses with a given square footage, what house prices are we likely to see? 
	*	Theoretical idea: impossible to compute *generation error*: requires every possible dataset in existence.

* *Test error*: like generalisation error but actually computable.

	* Basically, use *test error* to roughly approximate generation error.
	* Average loss on houses in test set: $$1/Ntest \sum_\limits{i=1}^{Ntest} L(y, f _\hat{w}(\mathbf{x}))$$
		* Note: $$ f _\hat{w}(\mathbf{x}) $$  was fit with training data.

* Defining overfitting:

	* Overfitting if there exists a model with estimated params $$w'$$ such that:
		1. Training error ($$\hat{w}$$) < Training error ($$w'$$).
		2. True  error ($$\hat{w}$$) > True error ($$w'$$)
	* In other words: overfit if training error is low compared to another model but "true error" is high. Better to have high training error but low "true error".

* Training/test split: how to think about dividing data between training and test split.
	*  General rule of thumb: just enough points in test set to approximate generalisation error well.
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
	* $$ Bias(x) = f_{w(true)}(\mathbf{x}) - f_\hat{w}(\mathbf{x}) $$	 
	* Bias is basically how good our fit is; more parameters generally means less bias.

* Variance:
	* How much variance does model have?
	* High complexity models == high variance.  
	* Variance is basically how wild our model is; more parameters means higher variance.

* Bias-variance tradeoff:
	* As error decreases, bias decreases and variance increases.
	* $$ MSE = bias^2 + variance $$ (mean-squared error)
	* Want to find sweet spot in model where bias and variance are together as low as possible.
	* We can't compute "true function" but there are ways to optimise an approximation a practical way.

* Error vs amount of data:
	* True error decreases as data points in training set increase to limit of bias + noise.
	* Training error goes up as data points increase to same limit.

* Validation set:
	* Choosing tuning parameters, $$ \lambda $$ (eg degree of polynomial):
		* Naive approach: For each model complexity $$ \lambda $$:
			1. Estimate params on training data.
			2. Assess performance on test data.
			3. Choose $$ \lambda $$ with lowest test
			* Problem with this the test data was already used for selecting $$  \lambda $$ - will be overfit.
		* Better approach: split data 3 ways: training, validation and test set
			* Select $$ \lambda $$ that minimizes error on **validation set**.
			* Get approximate generalisation error of $$  \hat{w}_{\lambda\star} $$   using **test set**.
			* Typical splits:
				* 80% / 10% / 10%
				* 50% / 25% / 25%
