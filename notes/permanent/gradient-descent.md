---
title: Gradient Descent
date: 2015-10-10 00:00
modified: 2024-04-04 00:00
summary: an optimisation algorithm to minimise a cost function
status: draft
---

Gradient descent is an optimisation algorithm used to minimise a cost function. It works by repeatedly calculating the [Derivative](../../../permanent/derivative.md) of the cost function with respect to the model's parameters and updating those parameters in the direction of the negative gradient.

In pseudocode, one step of Gradient Descent looks like this:

```
guess = some guess
d = derivate @ guess
cur guess = cur guess - alpha * d
```

Where `alpha` is some learning rate like 0.001.

For example, given a simple function $y = x^2$, we can use Gradient Descent to find the minimum.

If we plot the function, we can see that the minimum of the function is 0.

![](../../../_media/gradient-descent-y-x-squared.png)

We know from the [Power Rule](Power%20Rule) that the derivate of $x^2 = 2x$: $\frac{d}{dx} x^2 = 2x$.

So if we started with a guess of 5, one step of gradient descent would look like:

```
d = 2 * 5
5 - 0.001 * 10 = 4.99
```

Now our guess is at 4.9. A bit closer to 0. We can take another step:

```
d = 2 * 4.99 = 9.98
guess = 4.99 - 0.001 * 9.98 = 4.98
```

And now we're at 4.98. A bit closer again. If I run it for 10k steps and plot in red the guess at each 100th step, it looks like this:

![Gradient Descent Guess](../../../_media/gradient-descent-guess.png)

In Machine Learning, we'll typically compute the gradient with respect to the input features for every item in the dataset. When Gradient Descent is performed on a mini-batch (i.e. a subset of the data), it's referred to as [Stochastic Gradient Descent](stochastic-gradient-descent.md).
