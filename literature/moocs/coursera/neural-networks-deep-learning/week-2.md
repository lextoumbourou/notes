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

## Derivates with a Computation Graph

* Chain rule in calculus: if a affects v which affect J, a -> v -> J, then the amount J changes when a is nudged, is the product of how much v changes when a is nudged and now much J changes when v is nudged:

   ``dJ / da = (dJ/dv) * (dv/da)``

* Will be a final output variable you want to calculate `d FindOutputVar / d var`: generally represent it as `dvar`.
    * `dvar` - represent the derivative to final output with respect to various intermediate quantities. 

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

* [Speed examples](Vectorization.ipynb)

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

## Vectorizing Logistic Regression

* Instead of iterating through all training examples and calculating y_hat, can calculate as:
  $$ w^TX + B $$ (where B is a row vector containing just b and X contains all training examples).
* Written in Numpy as: ```Z = np.dot(w.T, X) + b```
  * Broadcasting in Python = convert a matrix to fit an equation.

## Vectorizing Logistic Regression's Gradient Output

* Original version of gradient descent with for loops:

    ```
    J = 0
    dw[1] = 0
    dw[2] = 0
    db = 0
    for i = 1 to m:
        z[i] = wT * x[i] + b
        a[i] = sigma(z[i])
        J += -(y[i] * log(a[i]) + (1 - y[i]) * log(1 - a[i]))
        dz[i] = a[i] - y[i]
        dw[1] += x[1][i] * dz[i]
        dw[2] += x[2][i] * dz[i]
        db ++ dz[i]
    
    J = J/m
    dw[1] = dw[1] / m
    dw[2] = dw[2] / m
    db = db / m
    ```

* Gradient descent using vectorization:

    ```
    Z = wT * X + b
    A = sigma(Z)
    dZ = A - Y
    dW = 1/m * X * dZT
    db = 1/m * np.sum(dZ)
    w := w - learning_rate * dw
    b := b - learning_rate * db
    ```

## Broadcasting in Python

* General principle:
  * If you have a (m, n) matrix and multiple by (1, n) matrix, it will turn into (m, n) matrix.