# Week 6: Nearest Neighbours & Kernel Regression

## Pre class notes:

* Class is about using "nonparametric" methods to fit data.
  * From Wikipedia: a non parametric method "does not require modeller to make any assumptions about the distribution of the population"
* Simplest example is "nearest neighbour regression".
  * Predictions for a single point is based on "most related" observations.
* Kernel regression is another example.
  * Similar to nearest neighbour but uses all observations weights by similarity to query point.

## Notes

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
	1. For some input, find closest $$ \mathbf{x}_i $$ using some distance metric.
	2. Predict: $$ \hat{y}_q = y_{nn} $$

* Distance metrics
	* 1D feature sets: Euclidian distance
		* $$ distance(x_j, x_q) = |x_j - x_q| $$ 
	* Multi dimensions:
		* Can use a weight on each feature to determine importance in computing distance ie sqft closeness is important, but year renovated may not be.
		* $$  distance(\mathbf{x_j}, \mathbf{x_q}) = \sqrt{a_1(x_i[1] - x_q[1])^2) + .. a_D(x_i[D] - x_q[D])^2)} $$   
			* Note: $$ a_D $$ is the "weight" in this equation.

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
