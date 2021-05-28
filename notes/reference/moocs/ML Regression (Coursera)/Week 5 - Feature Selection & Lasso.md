## Prerequisities

Matrix n-norm: https://www.youtube.com/watch?v=tXCqr2UsbWQ

## Notes

* Feature selection task motivation
	* Efficiency
		* when faced with large feature sets, predictions can get expensive.
		* When $$ \hat{w} $$ is sparse (eg has a lot of 0s in the dataset) can eliminate all 0 features.
	* Interpretability
		* which features are relevant to housing predictions?

* All subsets algorithm
	* Search over every combination of features computing the RSS: 
		* Iterate through every N = 1 sized combination, then N = 2 and so on up to feature D.
	* Slow to compute with a lot of features: complexity = $$ 2^{D} $$
		* With 15 features $$ 2^{15} = 32768 $$ 
 
* Greedy algorithms 
	* "Forward stepwise algorithm" 	
		* Find best N = 1 feature subset (maybe based on training error)
		* Next, find best N = 2 subsets that includes previous feature.
	* Use *validation set* or *cross validation* to choose when to stop greedy procedure - training error alone is insufficient.
	* Quicker to compute than all subsets, but may not find optimal combination.  
		* Complexity:
			* 1st step: D models, 2nd step: D - 1 models, 3rd step: D - 2 models etc	
			*  Worst case complexity: $$ D^2 $$
	* Other greedy algorithms
		* Backward stepwise: start with full model and work backwards removing one.
		* Combining forward and backward steps: add steps to remove features not considered relevant.

* Regularised regression for feature selection
	* Ridge regression (L2 penalty) encourages small weights but not exactly 0.
	* Core idea: can we use a similar idea but actually get weights to 0 and effectively remove them from the model?
	* Potential method 1: Thresholding.
		* Remove features that falls below some threshold for $$ w $$.
		* Problem: redundant features
			* If you have bathroom and number of showers, the weights would be distributed over both features. If you remove one, the coefficient should double on the other.
			* This means you could potentially end up discarding all redundant features. 
		* 
	* Potential method 2: use L1 norm for regularized regression aka "Lasso regression" aka "L1 regularised regression".
		* $$  RSS(\mathbf{w}) + \lambda * ||\mathbf{w}||_1 $$ 
			* When tuning param, $$ \lambda $$ is 0, $$ \hat{w}^{lasso} = \hat{w}^{LS} $$  (ie result is just least squares.
			* When $$ \lambda = \infty $$, $$ \hat{w}^{lasso} = 0 $$  (ie coefficients shrink to 0).
		* Coefficient path for Lasso:

			![Coefficient path for Lasso](coefficient-path-for-lasso.png)

			* For certain lambda values, features jump out of the model and other fall to 0. Individual impact of features becomes clearer.

* Optimising the lasso objective
	* The derivate of $$ |w_j| $$ can't be calculated when $$ |w_j| = 0 $$. Therefore, can't calculate the derivate.
	* Use "subgradients" over gradients. 

* Coordinate descent
	* Goal: minimise a function $$ g(w_0, w_1, ..., W_D) $$, a function with multivariables.
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
	
		$$ \underline{h}_j(\mathbf{x}_k) = \frac{h_j(\mathbf{x}_k)}{ \sqrt{\sum\limits_{i=1}^{N} h_j(\mathbf{x}_i)^2}} $$ 

* Coordinate descent for least squares regression
	* For each j feature, fix all other coordinates $$  \mathbf{w}_{-j} $$ and take the partial with respect to $$ \mathbf{w}_j $$
	* Gradient derived to $$ -2P_j + 2w_j $$ ($$ P_j $$ == residual without jth feature)
	* Setting to 0 and solving for $$ \hat{w}_j $$ results in $$ \hat{w}_j = P_j $$	
	* Intuition: if the residual between the predictions without j and the actual prediction is large, then the weight of j will be large and vice versa (to do: check this as your understanding improves).

* Coordinate descent for lasso (for normalised features)
	* Same as cord descent for least squares, except we set $$ \hat{w}_j $$ using a "thresholding" function (in code):

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
	* Normalisation factor is used during the set $$ w_j $$ portion of the regression.

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
	* Implement K-fold cross validation to select lasso tuning parameter $$ \lambda $$    

* Note: be careful about interpreting features, need to consider in the context of the entire model.
