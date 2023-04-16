---
title: "Improving Deep Neural Networks: - Week 3"
date: 2017-09-30 00:00
parent: deep-neural-network
category: reference/moocs
link: https://www.coursera.org/learn/deep-neural-network
status: draft
---

# Week 3

### Tuning process

* Order of important when hyperparam tuning:

    1. Alpha (learning rate) most important to tune.
    2. Beta (from momentum term)

    * 0.9 is a good default.

    3. Mini-batch size

    * Play around trying to speed up convergence.

    4. Number of layers (can sometimes make a huge difference).
    5. Learning rate decay.
    6. Rarely changes beta param in Adam.

* Don't perfom a grid search over hyper params, try random values - you'll end up searching more space.
* Consider using a "course to fine" search.

### Using an appropriate scale to pick hyperparameters

* Don't sample at random at uniform - pick the appropriate scale.
* If the hyper param is a integer on some small scale, then sampling uniformly at random may be appropriate.
* Trying to pick a value between 0.0001 and 1, want to search for hyper params on a log scale:
    * 0.0001, 0.001, 0.01, 0.1, 1
    * As follows:

        ```
        r = -4 * np.random.rand()  # A real number between -4 and 0.
        param = 10**(r)  # Random generate a number between 0.0001 and 1.
        ```

* Hyper params for exponentially weighted averages should also be sampled on a log scale.

### Hyperparameter tuning in practise: Pandas vs Caviar

* Hyperparam intuitions get stale - need to reevaluate occasionaly (once every several months).
* 2 schools of thought when searching for hyperparams:
    1. Babysit one model.
    * Overview:
        * Tune hyperparam and watch learning curve.
        * Tune another param and so on.
    * Use when you don't have a lot of compute capacity to train multiple models.
    * The "panda" approach, because pandas don't have many children.
    2. Train many models in parallel.
    * Overview:
        * Train a bunch of models in parallel and watch learning rate of each.
    * Use when you have a lot of compute capacity.
    * The "caviar" approach, because fish lay thousands of eggs and hope that a few will do well.

## Batch normalisation

### Normalising activations in a network

* Batch norm allows an algorithm to choose a much bigger range of hyper params and allows for a much bigger network.
* Basically, normalises the inputs from another hidden layer, rather than just the input layer, making it converge faster.
* Can either normalise a[l] (post activation values) or z[l] (pre activation), in practise batch norm is usually performed on z[l].
* Compute similarly to normalising inputs:

    $\mu=\frac{1}{m}\sum\limits_{i} z^{(i)} \\ \sigma^2=\frac{1}{m} \sum\limits_{i}(z_i-\mu)^2 \\ z_{norm}^{(i)}=z^{(i)} - \mu \\ z_{norm}^{(i)}=\frac{z^{(i)}}{\sigma^2 + \epsilon}$

    * Include the $\epsilon$ term in case sigma turns out to be 0.
* Don't want hidden units to always have mean 0 and variance of 1, so you introduce two learnable params $\beta$ and $\gamma$, which are added as follows:

    $z_{\text{~}}^{(i)}=\gamma * z_{norm}^{(i)} + \beta$

### Fitting Batch Norm into a neural network

* Apply batch norm at a neuron before computing activation function.
* Batch norm also has hyperparameters beta and gamma.
* Usually run on mini-batches, simply calculating the mean and std dev of just that batch.
* Since batch norm rescales by computing and subtracting the mean, the bias param will just be subtracted, so you don't need to include it in the implementation. Is basically replaced by the param $\beta$ that controls the scale of the normlisation.
* Dimensions of $\beta^{[l]}$ and $\gamma^{[l]}$ are $(n^{[l]}, 1)$.

### Why does Batch Norm work?

* Batch norm reduces the amount the hidden unit values jumps around when weights are updated.
* Batch norm also has a slight regularisation effect:
    * Each mini-batch is scaled by mean/variance calculated on just that mini-batch.
    * Adds noise to z[l] within that mini-batch.

### Batch Norm at test time

* Batch norm calculates mean/sigma for an entire mini-batch, but at test time, you may calculate for one example so need another approach.
* Estimate using an exponentially weighted average across each mini-batch across the training set.

## Multi-class classification

### Softmax Regression

* Used as activation function on last layer when performing multi-class classification.

    $t=e^{(z^{[l]})} \\ a^{[l]}=\frac{e^{(z^{[l]})}}{\sum\limits_{j=1}^{4}t_i}$

    ![Softmax example](/_media/softmax-example.png)

* Gives you a probability-like value (where all sum to one) for each category.

### Training a softmax classifier

* Called "softmax" because it returns a probability for each class rather than a hard yes or no.
* Generalisation of logistic regression two more than 2 classes.
* Loss for a single example:

    $y=[0, 1, 0, 0] \\ a^{[L]} = \hat{y} = [0.3, 0.2, 0.1, 0.4] \\ L(\hat{y}, y)=-\sum\limits_{j=1}^{4} y_j \ log \ \hat{y}_j$

    * Because you are summing over the log, the 0 values (wrong class) contribute nothing and the loss function ends up only taking the "activated" (1 value) class into consideration. High right weight = lower cost function.
* Loss for entire dataset:

    $J(w^{[i]}, b^{[i]})=\frac{1}{m} \sum\limits_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})$

## Introduction to programming frameworks

### Deep learning frameworks

* Criteria for choosing a framework:
    1. Ease of programming (development and deployment).
    2. Running speed: more are more efficient than others.
    3. Truly open (open source with good governance).

### TensorFlow

* Motivating problem, cost function to minimize:

    $J(w)=w^2-10w+25$

* Implemented as follows:

		import numpy as np
		import tensorflow as tf

		w = tf.Variable(0, dtype=tf.float32)
		cost = tf.add(tf.add(w**2, tf.multiply(-10, w)), 25)
		train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

		init = tf.global_variables_initializer()
		session = tf.Session()
		session.run(init)
		print(session.run(w))

		session.run(train)
		print(session.run(w))

		for i in range(1000):
		  session.run(train)

		print(session.run(w))

* Squaring, multiplication and adding syntax are overloaded, so you can replace:

		cost = tf.add(tf.add(w**2, tf.multiply(-10, w)), 25)

    with

		cost = w**2 - 10*w + 25

* To specify the values at run time, can utilise placeholders as follows:

		x = tf.placeholder(tf.float32, [3, 1])
		cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

		# ...

		coefficients = np.array([[1.], [-10], [25.]])

		# ...

		session.run(train, feed_dict={x:coefficients})
		print(session.run(w))

* These lines of code are idiomatic in TensorFlow:

		session = tf.Session()
		session.run(init)
		print(session.run(w))

    can be rewritten as:

		with tf.Session() as session:
		  session.run(init)
		  print(session.run(w))
