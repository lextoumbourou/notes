---
title: "Dust: pretraining transformers without backpropagation"
date: 2026-10-12 12:58
modified: 2026-10-12 12:58
status: draft
category: paper
summary: A zeroth-order method that pretrains small transformer language models from forward passes and noise, with no backward pass, and gets close to backpropagation.
tags:
- ZerothOrderOptimisation
- Backpropagation
paper_title: "Dust: Pretraining Transformers Without Backpropagation"
paper_url: https://qlabs.sh/research/dust
paper_authors: Samip Dahal, Bishwas Mandal, Serdar Gülbahar and Akshay Vegesna
paper_year: 2026
code: https://github.com/qlabs-eng/dust
aliases:
- Dahal et al., 2026
---

**Dust** is a method from Q Labs for pretraining a transformer language model without [Backpropagation](backpropagation.md). It estimates gradients from forward passes with added noise, which makes it a zeroth-order method. The authors call it the first zeroth-order method that is competitive with backpropagation at pretraining transformer language models [@dahalDustPretrainingTransformers2026].

It is a research page with a BibTeX entry, not an arXiv paper, and I found no peer-reviewed version when I read it on 12 October 2026.

## Method

Evolution strategies perturb the weights, so every member of the population needs its own perturbed copy of the network and its own forward pass [@sarkarEvolutionStrategiesHyperscale2025]. Dust perturbs activations instead.

1. Add Gaussian noise to the output of each linear layer, independently at every token. Each token is a separate experiment, so one forward pass evaluates thousands of them. The authors call this a *virtual population*.
2. Reward each token's noise by the change in loss at that token.
3. Average the reward-weighted noise over many draws. This estimates the error at the layer's output.
4. Multiply that estimated error by the layer's input to get the weight gradient.

Backpropagation also builds the weight gradient from an error and the layer's input. The only difference is where the error comes from: the chain rule there, a population of noisy trials here. Attention internals get a variant, because a token's noise also changes later tokens.

## Results

The authors train GPT-style models on FineWeb with SGD with momentum. The base model has 8 layers and width 512. Token budgets run from 100k to 20M, and populations from 64 to 16k draws.

- At 1M tokens, Dust ends below backpropagation from about a thousand draws.
- At 10M tokens, the 8-layer model reaches a test loss of 5.036 with 16k draws, against 5.015 for backpropagation.
- Even with 256 times the population, weight-space evolution strategies (the EGGROLL transformer) do not reach Dust at 64 draws. The authors estimate Dust is 10^3 to 10^4 times more efficient, which is an extrapolation.
- A 243M-parameter model beats one 120 times smaller at most population sizes, against the usual belief that zeroth-order methods get worse as networks grow.
- The estimated gradients line up better with backpropagation's as the population grows, but never match it exactly. The authors say this is useful, because the different direction gives a different path through the loss landscape.

## Caveats

The authors write that they "do not attempt to make it compute-efficient enough to replace backprop today". Every update uses thousands of draws, the models are small and the training runs are short. The claim that Dust could eventually beat backpropagation rests on a curve fitted to results that are still improving, which the authors call loosely constrained.

## Related

Their motivation is [The Bitter Lesson](https://www.incompleteideas.net/IncIdeas/BitterLesson.html) [@suttonBitterLesson2019]: general methods that scale with compute eventually win. MeZO showed that forward passes alone can fine-tune language models of up to 66B parameters [@malladiFineTuningLanguageModels2023], but not pretrain them. See also [Gradient Descent](gradient-descent.md).
