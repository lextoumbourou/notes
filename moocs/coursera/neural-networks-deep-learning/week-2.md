# Logistic Regression as a Neural Network

## Binary Classification

* Logistic Regression: algorithm for binary classification.
* Compute represents image as 3 matrices of width x height, with each matrix representing the red, green or blue colour intensity eg `3 * 64x64`
  * For ML, you want to conver  this into a single feature vector.
  * Convert to one vector of length: `n_x = 64 x 64 x 3 = 12288`
  * Refer to number of training examples as `m`.

## Logistic Regression

* Given x, want $$ \hat{y} = P(y = 1 | x)$$
  * "given some image, output a prediction"
* Params: $$ x \in \Re^{n_x}, b \in \Re $$
  * To get "probability" as output, use the sigmoid function: $$ \hat{y} = \sigma(w^{T}_{x} + b) $$
    * Big inputs to sigmoid get close to 1, large negative numbers, get close to 0:

        ```
        >> sig = lambda z: 1./(1+math.e**(-z))
        >> sig(-12)
        6.1441746022147215e-06
        
        >> sig(-1)
        0.2689414213699951
        
        >> sig(2)
        0.8807970779778823
        
        >> sig(5)
        0.9933071490757153
        ```

## Logistic Regression Cost Function

* To train W and b (weights and bias/intercept), you need to define a cost function.
* Typical loss function, squared error: $$ \frac{1}{2}(\hat{y} - y)^2 $$
  * Doesn't work well for gradient descent:  optimisation becomes non convex: has multiple local minima.
* Another loss function: $$ -(y \log \hat{y} + (1 - y) \log (1-\hat{y})) $$
  * If $$ y = 1 $$, want $$ \log \hat{y} $$ large, therefore, want $$ \hat{y} $$ large.
  * If $$ y = 0 $$, want $$ \log 1 - \hat{y} $$ large, therefore, want $$ \hat{y} $$ *small*. 
* Cost function: measures how well you're doing on the entire set:
  * $$ J(w, b) = \frac{1}{m} \sum\limits^{m}_{i = 1} L(\hat{y}^{(i)}, y^{(i)}) $$
* Logistic regression can be viewed as a "very small neural network".

## Gradient Descent

* Partial derivate symbol: $$ \partial $$, just means you're calculating the derivate for a function with multiple inputs.

## Derivatives

* Derivate is simply, if you nudge a value on the x variable in a certain direction, how much impact does it have on the y variable?

## More Derivative Examples

* Derivate of a^2 = 2a.

# Python and Vectorization

## Vectorization

* "The art of getting rid of explicity for loops in your code."
* Non-vectorized calculation of $$ z = {w^T}_x + b $$:

    ```
    z = 0
    for i in range(n - x): 
        z += w[i] * x[i]
    x += b
    ```

* Vectorized:

     ```
     np.dot(w, x)
     ```

* [Speed examples](./notebooks/Vectorization.ipynb)

## More examples of vectorization

* Applying an exponent to every element:

    ```
    import numpy as np
    u = np.exp(v)
    ```

* Take the log of every element:

    ```
    import numpy as np
    u = np.log(v)
    ```
