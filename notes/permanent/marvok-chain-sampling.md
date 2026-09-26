---
title: Markov Chain Sampling
date: 2024-10-03 00:00
modified: 2026-09-26 08:55
status: draft
---

**Markov Chain Sampling**, usually called Markov Chain Monte Carlo (MCMC), is a family of methods for drawing samples from a probability distribution that's hard to sample from directly.

It works by building a [Markov Chain](markov-chain.md) whose stationary distribution is the target distribution, then running the chain for many steps. After enough steps, the states it visits approximate samples from the target distribution.

Common examples are the Metropolis-Hastings algorithm and Gibbs sampling. Consecutive samples are correlated, and the early samples are usually thrown away (the burn-in period) before the chain settles.
