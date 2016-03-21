# Week 2


## Data likelihood

* Quality metric for logistic regression
* Want probability to be close to correct classification for some $$ \hat{w} $$ and features.
* Data likelihood = Calculate probability that result equals the actual output for each data point in the training dataset then take product of results:

	$$ l(w) = P(y_1 | \mathbf{x}_1,w) * P(y_2 | \mathbf{x}_2,w) ... * P(y_N | \mathbf{x}_N,w) $$

	$$ l(w) = \prod\limits_{i=1}^{N} P(y_i | \mathbf{x}_i, \mathbf{w}) $$

	* Note: you multiply the probabilities because you assume probabilities of each data point are independent.

## Review of gradient ascent

* Algorithm for one-dimensional space:

	while not converged:
    $$ w^{(t+1)} < w^{(t)} + \eta \frac{dl}{d} $$

* Rough intuition: step towards a maximum multiplying the derivative by a step size, until the derivate is basically 0, which means you have hit the maximum. If you over step, then you should be pulled in the opposite direction, meaning you should eventually converge.

* When multi-dimension, replace derivative with a D-dimensional vector of partial derivatives:

$$ \triangledown l(\mathbf{w}) =\begin{bmatrix}\frac{\partial l}{\partial w_0}\\\frac{\partial l}{\partial w_1}\\\frac{\partial l}{\partial w_D}\end{bmatrix} $$

## Learning algorithm for logistic regression

Derivative of (log-)likelihood:

$$ \frac{\partial l(\mathbf{w})}{\partial \mathbf{w}_j} = \sum\limits_{i=1}^{N} h_j(\mathbf{x}_i)(\mathbf{1}[y_i = +1] - P(y = +1 | \mathbf{x}_i, \mathbf{w})]]) $$

Rough attempt at writing gradient ascent:

```
#...

converged = False
while not converged:
   for j in range(len(weights)):
       partial[j] = (d[j] * (get_indicator(data) - get_probability(data, coefficients)).sum()
   coefficients = step_size * partial
   converged = assess_convergence_somehow(partial)  # ??
```

## Interpreting derivative for logistic regression

* Contribution for a single datapoint should work as follows:
	* When probability output is close to 1, the partial for that coefficient should be close to 0:
		* For example, if the probability is 1, the partial would be calculated like: (1 - 1) = 0
			* If the probability was way off, it would be closer to 1 and push the coefficient value up.
		* On the other hand, if it's a negative datapoint and we get it wrong: (0 - 1) = -1 which should push the coefficient in the negative direction.
			* If we get it right, the probability of it being positive should be near 0, so: (0 - 0), which would keep the coefficient the same.

## Choosing the step size η

* Too small step size: too many iterations until convergence.
* Too high step size: overstep the convergence point.

## Rule of thumb for choosing step size

* Requires lots of trial and error: hard.
* Try several values, exponentially spaced:
	* Find one that's too small.
	* Find one that's too large.
	* Try values inbetween.
