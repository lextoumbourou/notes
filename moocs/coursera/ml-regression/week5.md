# Week 5 - Feature Selection & Lasso

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

			![Coefficient path for Lasso](./images/coefficient-path-for-lasso.png)

			* For certain lambda values, features jump out of the model and other fall to 0. Individual impact of features becomes clearer.