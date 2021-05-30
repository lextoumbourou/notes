---
title: "Deep Learning for Coders with Fastai and Pytorch: AI Applications Without a PhD"
date: 2021-05-30
status: draft
tags:
  - Deep Learning 
---

## Chapter 1. Your Deep Learning Journey

* Deep learning is for everyone: doesn't require lots of maths, lots of data or lots of expensive compute, as people once had us believe.
* It's a computer technique that extracts and transforms data using multiple layers of neural networks.
* Neural networks history:
	* 1943 Warren McCulloch (neurophysiologist) and Walter Pitts (logician) developerd math models of artificial neuron with paper “A Logical Calculus of the Ideas Immanent in Nervous Activity,”
		* They simplied a model of a real neuron with simple addition and thresholding.
	* Work was taken up by Frank Rosenblatt who developed the Mark I Perceptron in “The Design of an Intelligent Automaton”
	* Another book was later written about the invention called: Perceptrons (MIT Press) 
	* Perhaps the most pivotal work in neural networks in the last 50 years was the multi-volume Parallel Distributed Processing (PDP) by David Rumelhart, James McClelland,
		* PDP approach very similar to today's approach.
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