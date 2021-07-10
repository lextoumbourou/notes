---
title: "Deep Learning for Coders with Fastai and Pytorch: AI Applications Without a PhD"
date: 2021-05-30
status: draft
tags:
  - DeepLearning 
---

## Chapter 1. Your Deep Learning Journey

* Deep learning is for everyone: doesn't require lots of maths, lots of data or lots of expensive compute, as people once had us believe.
* It's a computer technique to extract and transform data using multiple layers of neural networks.
    * Each layer takes input from previous layers and refines them.
    * Layers trained to minimise errors


* Neural networks history:
	* 1943
        * Mathematical model of artificial neuron is described in "A Logical Calculus of the Ideas Immanent in Nervous Activity" by Warren McCulloch (neurophysiologist) and Walter Pitts (logician)
            * Simplified model of a brain's neuron could be represented with addition and thresholding.
                * Thresholding example: `value = max(0, value)`
            * Walter Pitts was self-taught and homeless most of his life.
    * 1958 
        * Work was later up by Frank Rosenblatt who developed the Mark 1 perceptron machine describe in “The Design of an Intelligent Automaton”
    * 1969
        * Perceptrons (MIT Press) by Marvin Minsky and Seymour Papert showed a single layer of neurons could not learn functions including XOR
            * However, same book also showed 2 layers could but that was larger ignored
    * 1986
        * Pivotal work in neural networks was release: multi-volume Parallel Distributed Processing (PDP) by David Rumelhart, James McClelland.
            * PDP approach very similar to today's approach with requirements:
                * A set of processing units
                * A state of activation
                * An output function for each unit
                * A pattern of connectivity among units
                * A propagation rule for propagating patterns of actvities through the network of connectivities
                * An activate rule for comping the inputs impinging on a unit with the current state of unit to product an output
                * A learning rule where patterns of connectivity is modified by experience
                * An environmnet within which system operates
	* Neural networks widely used in the 80s and 90s, but they weren't able to get very deep (adding more than 2 layers) without breaking or being too slow.
		* Only recently we've learned the power of neural networks with more compute, bigger datasets and some tweaks (like Dropout).
	* How to learn Deep Learning
		* Harvard professor David Perkins, who wrote Making Learning Whole (Jossey-Bass), teach the "whole game"
			* Baseball example: That means that if you’re teaching baseball, you first take people to a baseball game or get them to play it. You don’t teach them how to wind twine to make a baseball from scratch, the physics of a parabola, or the coefficient
			* In Deep Learning: helps if you have the motivation to fix model 
		* Hardest part of DL is "artisanal": learning if your model is training properly, have you got enough data etc
	* Modern deep learning terminology
		* "Functional form" of model is architecture (people use model to mean architecture confusing)
		* Weights are parameters
		* Predictions are calculated from independant variables, data including the labels
		* Results of models are predictions
		* Measure of performance is loss
		* Loss depends on predictions but also correct labels

## Chapter 2

* [[Out-of-domain data]] (pg. 104)
    * Data that is given to a model that is different to the data it was trained on.
    * No complete technical solution: have to carefully roll out model
* [[Domain Shift]] (pg. 104)
    * Data that model performs inference on changes over time.
    * Example: types of customers an insurance sees changes over time, making their earlier models less useful.
    
    ## Chapter 4. Under the Hood: Training a Digit Classifier
    
    * [[Exponential Function]]
        * Defined as $e^x$
            * $e$ is a number about 2.718
                * The inverse of natural logarithm
        * Always positive and increases fast
    * [[Softmax Function]] (pg. 223-227)
        * The multi-category equivalent of the [[Sigmoid Function]]
            * Similarly "smooth and symmetric" properties
        * Use if more than 2 categories and want probabilities add to 1.
            * Can also use when there are just 2 categories to be consistent
        * If one of the numbers is slightly bigger, exponential amplifies it
        * Softmax "wants" to pick a single result.
            * Use model with multiple binary columns to handle items model hasn't seen.
        * One part of the [[Cross-Entropy Loss Function]]
    * [[Log Likelihood]] (pg. 226-231)
        * For a vector of softmaxed predictions, take the prediction that corresponds with the correct label.
        * Then apply `-log(prediction)`
            * In PyTorch, log uses $e$ as the base.
            * Note that because a log between 0 and 1, it has a negative log value. We invert it with negative.
            * The closer to 1, the closer to 0 loss.
        * In PyTorch, the `nll_loss` function doesn't take the log. It expects it to be already taken.
    * [[Cross-Entropy Loss Function]]
        * Defined as:  [[Softmax Function]] then negative [[Log Likelihood]]
        * Consider gradient of `cross_entropy(a, b)` is `softmax(a)-b`
        * When `softmax(a)` is final activation, gradient is the same as diff between prediction and target
            * So it's the same a [[Root mean-squared error - L2 Loss]] in regression.
            * Because gradient is linear, don't see sudden jumps or exponential increases in gradients