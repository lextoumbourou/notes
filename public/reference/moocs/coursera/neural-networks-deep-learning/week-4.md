# Week 4

## Deep L-layer neural network

* Refers to neural networks with large hidden layers.
* $L$ denotes number of layers in network in course notation.
* $n^{[l]}$ = number of units in layer $l$.
* $a^{[l]}$ = number of activations in layer $l$.
* $W^{[l]}$ = weights for $z^{[l]}$.

## Getting your matrix dimensions right

* $z^{[1]} = W^{[1]} * x + b^{[1]}$
  * z = number of activations for first hidden layer = `(3, 1)` or $(n^{[1]}, 1)$ dimensional vector.
  * x has 2 input features, so it's a $(2,1)$ dimensional vector.
    * $W^{[1]}$ therefore needs to result in a $(n^{[1]}, 1)$ dimensional vector.
    * `(3, 2)` matrix * `(2, 1)` vector = `(3, 1)` vector.
    * (missing notes)

* General formula: $W^{[l]} = (n^{[l]}, n^{[l-1]})$

## Why deep representations?

* Intuition about deep representation:
  * Image example:
    * First layer may find edges by grouping together pixels.
    * Second layer may group edges and start to detect parts of faces.
    * Third layer may group parts of faces to start to detect different faces.
  * Audio example:
    * Audio -> phonemes -> words -> sentences.

  * Multiple XORs would require an exponentially large set of neurons for one layer.

## What are hyperparameters

* Parameters: W[1], b[1], W[2], b[2], W[3], b[3]
* Hyperparameters:
  * Learning rate
  * Iterations
  * Hidden layer count.
  * Hidden unit count.
  * Choice of activation function.
  * Momentum.
  * Minibatch size
* "Applied deep learning is a very empirical process".
  * "Try out a lot of things and see what works"

## What does deep learning have to do with the human brain?

* Punchline: "very little".
* A biological neuron takes electrical signals from other neurons and does a threshold calculation and sends a electrical pulse to other neurons if it fires.
* "Neuroscientists have almost no idea what a single neuron is doing"

## Exercise

* 70% on attempt one. Need to review the lesson on dimensions.
