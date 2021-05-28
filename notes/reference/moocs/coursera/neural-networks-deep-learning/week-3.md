# Week 3

## Neural Networks Overview

* Neural networks adds another layer of computation called the "hidden layer".
  * In a one-layer, NN, hidden layers is simply another set of weights that take all inputs as features.
  * Final layer: output layer. It is analogous to a simple logistic regression model.

## Neural Network Representation

* Input layer referred to as $a^{[0]} = x$.
  * $a$ is used to refer to "activations" of the layer.

* Hidden layer referred to as $a^{[1]}$.
  * Each computational node referred to with subscript notation: $a_{1}^{[0]}$
  * Called a "hidden" layer because you don't see the layer as part of the training set.
  * The entire layer in this example is a 4 dimensional column vector:

    ![](assets/week-2638186e.png)

* Output layer referred to as $a^{[2]} = \hat{y}$
* Input layer is not counted when counting layers, example in the video is considered a 2-layer NN:

    ![](assets/week-c2e36574.png)

## Computing a Neural Network's Output

* "Like logistic regression but repeated a lot of times".
* Each node in the hidden layer computes sigmoid of the weighted sum of input features and weights for all inputs (logistic regression just has a single node).
* Equation for a single neuron: $z_1^{[1]} = w_1^{[1]}T x + b_1^{[1]}, a_1^{1]} = \sigma({z_1^{[1]}})$
* You can then calculate the entire $z$ as a vector by transposing the column vectors into row vectors and stacking into a matrix. Can then multiple by the input features as follows:

  ![](assets/week-bc0eaea0.png)

* Entire algorithm:

  $$
  z^{[1]} = W^{[1]}x + b^{[1]} \\
  a^{[1]} = \sigma(z^{[1]}) \\
  z^{[2]} = W^{[2]}a^{[1]} + b^{[2]} \\
  a^{[2]} = \sigma(z^{[2]}) \\
  $$

## Vectorizing across multiple training examples


* Algorithm with a loop:

  $$
  \text{for i = 1 to m,} \\
  z^{[1]} = W^{[1]}x + b^{[1]} \\
  a^{[1]} = \sigma(z^{[1]}) \\
  z^{[2]} = W^{[2]}a^{[1]} + b^{[2]} \\
  a^{[2]} = \sigma(z^{[2]}) $$
  $$

* To compute without a loop:

  $$
  z^{[1]} = W^{[1]}X + b^{[1]} \\
  a^{[1]} = \sigma(z^{[1]}) \\
  z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\
  a^{[2]} = \sigma(z^{[2]}) $$
  $$

* For the `X`s, it's features stacked vertically and different training examples horizontally.
* For the `Z`s and `A`s, a single matrix refers to a single layer. Vertically they different hidden units and horizontally they refer to training examples.

  ![](assets/week-3f25f74e.png)

## Activation functions

* `tanh` function is generally superior to a `sigmoid` activation.

  * Except in the case of binary classification (hot dog or not ☺).
  * Formula for `tanh`:

    $$ \text{tanh}(z) = \frac{e^{z}-e^{-z}}{e^z+e^{-z}} $$

* Activation functions can be different for different layers.

* Downside of `sigmoid` and `tanh`: if z is large or small, slope can be small slowing down gradient descent.

  * Popular alternative: `relu` function.

* `relu` activation function:

  $$ \text{relu}(z) = max(0, z) $$

  * If you don't know what to use, just use `relu`: most people use it.

* One disadvantage to `relu` is the derivate is equal to 0 when z is negative. Alternative: leaky relu, which is the max between z and z * some very small number:

  $$ \text{leaky_relu(z) = max(0.001 * z, z)}

## Why do you need non-linear activation functions?

* "To compute interesting functions, you do need to pick a non-linear activation function".
* If you used a linear activation function, no matter how many layers you have, you're just creating linear activation function, so you might as well not have any layers.

## Derivatives of activation functions

## Gradient descent for Neural Networks

## Backpropagation intuition

## Random initialization

* For a neural network, you need to initialize weights randomly.

## Exam

* To do:
  1. Rewatch random init video.
  2. Rewatch video and try to determine what the shapes of the matrices should be.
