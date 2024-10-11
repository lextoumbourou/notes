---
title: "Improving Deep Neural Networks - Week 1"
date: 2017-09-30 00:00
parent: deep-neural-network
category: reference/moocs
link: https://www.coursera.org/learn/deep-neural-network
status: draft
tags:
    - DeepLearning
---

# Week 1

## Settings up your Machine Learning application

### Train / Dev / Test sets

* NN require a lot of decisions:
    * How many layers?
    * How many hidden units?
    * What learning rate?
    * What activation functions?
* ML is a highly iterative process: start with idea and keep iterating.
* Usually take data and split into 3 sets:
    * Training
    * Hold-out, cross val, dev set.
    * Test set.
* Traditional ML: 60%/20%/20% split when dealing with 100, 1000 or 10000 items in the set..
* Neural nets tend to have much larger datasets (1m+), so split may be more like: 98%/%1/1%, with 10000 items in the dev set.
    * Want a dev set that let's you quickly iterate through algorithms to figure out quickly which ones are better.
* May be training on a mismatched train/test set distribution.
    * Dev/test sets may have pictures uploaded by users.
    * Train may have pictures downloaded from the internet.
    * Should ensure dev/test sets are from the same distribution.
* May not need a test set (only dev set), but won't give you an unbias estimate of performance (maybe don't need one?).

### Bias / Variance

* High bias: underfitting.
    * Doesn't perform well on training set, generally won't perform well on dev set.
* High variance: overfitting.
    * Performs well on training set but poorly on the dev set.
* Accuracy of model should be compared to human accuracy: optimal (bayes) error.
    * If human's achieve 15% error, then a model that achieve similar would be considered low bias, low variance.

### Basic recipe for Machine Learning

* Basic recipe:

    1. After training model, ask "does the model have high bias?"
        * Try bigger network.
        * Train longer.
        * Try a different NN architecture (maybe).
    2. Do you have high variance?
        * More data.
        * Regularisation.
        * Different NN architecture.
* "Trade off" of bias / variance is talked about less in the deep learning era, because there are things you can do that reduce one without increasing the other.
    * Increasing data improves variance without bias.
    * Bigger networks let you decease bias without hurting variance.

## Regularising your neural network

### Regularisation

* L2 regularisation = $||w||^{2}_{2} = \sum\limits_{j=1}^{n_x} w_j^{2} = w^Tw$
	* Added to cost function definition in linear regression as follows: $J(w, b) = 1/m \sum\limits_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} ||w||^{2}_{2}$
		* Where $\lambda$ controls the amount of regularisation in the model.
		* A value of 0 = no regularisation.
		* A value of ∞ = only regularisation.
		* Generally only regularise the weights, not the bias term.
* L1 norm = $\frac{\lambda}{2m}||w||_ {1}$
    * L1 norm will generally make the weights sparse, allowing you to compress the model, though doesn't always work well in practise.
* L2 norm in NNs, refers to the sum of the weight matrix or norm of the matrix:
    * $J(w^{[1]},b^{[1]},w^{[L]},b^{[L]})=\
  \frac{1}{m} \sum\limits_{i=1}^{n} L(\hat{y}^{(i)}, y^{(i)}) +\
  \frac{\lambda}{2m} \sum\limits_{l=1}^{L}||W{[l]}||^2$
        * "Squared norm is the sum of the i sum of j, of each of the elements of that matrix, squared."
            * $||W^{[l]}||_ {F}^{2} = \sum\limits_{i=1}^{n[l-1]} \sum\limits_{i=1}^{n[l]} (w_{ij}^{[l]})^2$
            * The L2 norm of a matrix is actually called the "Frobenius norm".
* The only change to the backprop step is adding the weight as follows:
    $dw^{[l]} = \text{(from backprop)} + \frac{\lambda}{m}W^{[l]} \\ W^{l} := W^{[l]} - α dW^{[l]}$

### Dropout Regularization / Understanding Dropout

* Define a prob that a hidden unit will be kept. Eg keep_prob = 80%:
    * 20% of nodes will be set to 0.
* Don't use drop out at test time.
* Can't rely on a single feature so they tend to spread out weights.
    * Generally shrinks weights.

### Other regularisation methods

* [Data Augmentation](../../../../permanent/data-augmentation.md)
    * Flip images, random zoom, crop, etc.
    * Not as good as new data.
* Early stopping.
    * As you train longer, the training error tends to decrease but the dev tends to increase. Stop early before dev error increases.
    * Can make compartmentalisation of each tasks hard: want to optimise, then reduce overfitting etc.
        * Early stopping couples optimisation and overfitting.
        * Better to just use L2 regularisation.

## Setting up your optimisation problem

### Normalising inputs

* Normalising inputs corresponds to 2 steps:
    1. Subtract out the mean, so you training set has a mean of 0:
       $\mu = \frac{1}{m} \sum\limits_{i = 1}^{m} x^{(i)} \\ x := x - \mu$
    2. Normalise variance:
       $\sigma^2 = \frac{1}{m} \sum\limits_{i=1}^{m} x^{(i)} ** 2$
       _ $** 2$ refers to element-wise squaring.
       _ Then, take each example and divide it by sigma squared: $x /= \sigma^2$
* Use the calculated mu and sigma squared to normalise test set.
* Unnormalised inputs will require a smaller learning rate.
* Normalised inputs will usually make cost function faster to optimise.

### Vanishing / Exploding gradients

* Derivates can get very big (exploding) or very small (vanishing).
* If you initialise your weights to 1.5 \* identity matrix (or some value that would cause it to grow exponentially), the value of $\hat{y}$ would explode.
* The inverse is also true: very small weights, could cause $\hat{y}$ to vanish.
* This can cause gradient descent to take a long time to run.

### Weight initialization for Deep Networks

* Larger n is (eg the larger the number of inputs is), the small you want each of the corresponding weights to be.
* One approach: set variance of $w_i$ to be 1/n: $\text{var}(w_i)=\frac{1}{n}$
    * When using relu, generally 2/n works better.
    * tanh activation, prefers $\sqrt{\frac{1}{n^{[l-1]}}}$.
    * Also, Xavier init method: $\sqrt{(\frac{2}{n^{[l+1]} + n^{[l]}})}$
* Can be set as follows:
    $W^{[l]}=\text{np.random.randn(shape) * np.sqrt}(\frac{2}{n^{[l-1]}})$ \* Where the n for each layer refers to the size of the previous layer's output.
* Doesn't solve, but helps reduce the vanishing/exploding problem.

### Numerical approximation of gradients

* Gradient checking: tool that can help determine if an implementation of backprop is correct.
* Numerical approximation of gradients:

    1. Start with function: $f(\theta)=\theta^3$
    2. Instead of just nudging it in one direction to get an estimate of the gradient, can do it in two: $f(\theta + \epsilon)$ and $f(\theta + \epsilon)$
    3. Make use of both when calculating the derivative, diving by the width of the slope and it should be close to $g(\theta)$: $\frac{f(\theta + \epsilon) - f(\theta + \epsilon)}{2\epsilon} \approx g(\theta)$

    * By setting $f(\theta)=\theta^3$ and $theta=1$ and $\epsilon=0.01$, it works as follows:
		* $\frac{(1.01)^3 - (0.99)^3}{2(0.001)}$

				(1.01 ** 3 - (0.99)**3) / 2 * (0.001)
				>> 3.0001000000000057e-05

    * $g(\theta)=3\theta^2=3$
    * Then you get an approx error of 0.0001.
* Using this gradient method is twice as slow and generally will only be used in debug not in training.
* Summary: 2-sided difference formula is much more accurate than the one-sided.

### Gradient checking

* Take all params $W$ and $b$ and reshape into a vector then concatenate into a vector $\theta$.
* Do the same for all derivatives $dW$ and $db$ into vector $d\theta$.
* $\text{for each i:} \\d\theta_{\text{approx}}^{i} = \frac{J(\theta_1, \theta_2, ..., \theta_i + \epsilon) - J(\theta_1, \theta_2, ..., \theta_i - \epsilon)}{2\epsilon}$
    * The result should be close to the derivative (take the eclidean distance of both vectors to confirm): $d\theta_{approx} \approx d\theta$
    * If the euclidean distance is $10^{-7}$ off, then it's probably correct. At $10^{-3}$, it's probably incorrect.

### Gradient checking implementation notes

* Don't use in training, only while debugging.
    * Train a few iterations with grad check to see if correct then turn off.
* If an algorithm fails grad check, want to look at individual values to try to figure out what's up.
* Remember to include regularization term.
* Won't work with dropout.
