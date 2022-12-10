# Abstract

* The study on improving the robustness of deep neural networks against adversarial examples grows rapidly in recent years.
* Among them, adversarial training
is the most promising one, which flattens the input loss landscape (loss change
with respect to input) via training on adversarially perturbed examples
However,
how the widely used weight loss landscape (loss change with respect to weight)
performs in adversarial training is rarely explored. 
* In this paper, we investigate the weight loss landscape from a new perspective, and identify a clear correlation
between the flatness of weight loss landscape and robust generalization gap. Several well-recognized adversarial training improvements, such as early stopping,
designing new objective functions, or leveraging unlabeled data, all implicitly
flatten the weight loss landscape
* Based on these observations, we propose a simple
yet effective Adversarial Weight Perturbation (AWP) to explicitly regularize the
flatness of weight loss landscape, forming a double-perturbation mechanism in the
adversarial training framework that adversarially perturbs both inputs and weights.
* Extensive experiments demonstrate that AWP indeed brings flatter weight loss
landscape and can be easily incorporated into various existing adversarial training
methods to further boost their adversarial robustness.

# Introduction

h deep neural networks (DNNs) have been widely deployed in a number of fields such as
computer vision [13], speech recognition [47], and natural language processing [10], they could be
easily fooled to confidently make incorrect predictions by adversarial examples that are crafted by
adding intentionally small and human-imperceptible perturbations to normal examples [

As DNNs penetrate almost every corner in our daily life, ensuring their security, e.g., improving their
robustness against adversarial examples, becomes more and more important.

There have emerged a number of defense techniques to improve adversarial robustness of DNNs
[33, 25, 48]. Across these defenses, Adversarial Training (AT) [12, 25] is the most effective and
promising approach, which not only demonstrates moderate robustness, but also has thus far not been
comprehensively attacked [2]. AT directly incorporates adversarial examples into the training process
to solve the following optimization problem:

