---
title: Genetic Algorithms
date: 2025-09-07 00:00
modified: 2025-09-07 00:00
category: permanent
summary: "an optimisation technique inspired by natural selection"
aliases:
- Genetic Algorithm
tags:
- EvolutionaryAlgorithms
cover: /_media/genetic-algorithms-cover.png
hide_cover_in_article: true
---

**Genetic Algorithms** are an optimisation technique inspired by natural selection.

The high-level idea is that we start from a population of possible solutions where we have a way to evaluate how good each solution is - called a **fitness function**. Then we perform a loop that involves selecting from the fittest solutions (with randomness), breeding these solutions together, creating mutations, and replacing the population with the new solutions.

In biology, all living organisms have a genetic code that stores their information, called DNA. A genetic code is called a [Genotype](genotype.md), and the expression of it (i.e. a living human) is called a [Phenotype](phenotype.md). Genetic algorithms borrow this terminology, and typically a genotype is encoded as a string, array, or graph, where each element is a gene. Then, when we expand the genotype into a solution (for example, creating a robot in a simulation), we call it a phenotype.

The algorithm looks like this:

1. Create a random population of candidate solutions (likely encoded as an array).
2. Evaluate the fitness of each individual in the population (see **Fitness Function**).
3. Select "parents" based on their fitness (see **Selection**).
4. Parents create a new "offspring" (see **Crossover**).
5. Offspring is mutated in some way with some probability (see **Mutation**).
6. Replace some or all of the population with the new offspring (see **Replace**).
7. Repeat 2-6 until the termination condition is met.

<a href="_media/genetic-algorithms-overview.png" target="_blank">
  <img src="_media/genetic-algorithms-overview.png" alt="Genetic Algorithms Overview - an overview of this article in visual form"  style="max-width: 100%" />
</a>

One of the most famous examples comes from [Evolving Virtual Creatures](../../../permanent/evolving-virtual-creatures.md) by Karl Sims. It's also recently been used within [AlphaEvolve](../reference/papers/alphaevolve-a-coding-agent-for-scientific-and-algorithmic-discovery.md), which developed novel solutions to many problems by utilising techniques of Genetic Algorithm, utilising LLMs for the breeding solutions together.

## [Fitness Function](../../../permanent/fitness-function.md)

A fitness function evaluates the population in some way. In the Virtual Creatures example, the fitness is based on how far they travel under different conditions. In AlphaEvolve, the fitness is a test specific to the problem it is trying to solve, designed to find an optimal solution.

## [Selection](../../../permanent/selection.md)

Selection algorithms determine which individuals will become parents for the next generation, giving fitter individuals a higher chance of reproduction while maintaining some diversity.

### Roulette Wheel

When selections are based on the proportion of fitness across the population, think about a roulette wheel as a pie chart, where each pie is proportional to its fitness as a whole. The more fit take up more space, so they're more likely to be selected, but there's also some entropy to ensure the solution space is explored.

1. Calculate the fitness of all individuals
2. Generate a random number between 0 and the whole fitness.
3. Select the individual whose cumulative fitness exceeds that of others.

### Tournament Selection

A simpler approach, where we select some individuals at random from the population and choose one of the highest-fitness individuals.

### Rank Selection

Individuals are ranked by fitness, and the selection probability is based on rank rather than raw fitness values. This approach is particularly helpful when fitness values exhibit large variations or when dealing with negative fitness values.

## [Crossover](crossover.md)

### One-Point Crossover

A single crossover point is chosen randomly. Everything before this point comes from Parent 1, and everything after comes from Parent 2.

For example:

- Parent 1: `[1, 0, 1, 1, 0, 0, 1]`
- Parent 2: `[0, 1, 0, 0, 1, 1, 0]`
- Child: `[1, 0, 1 | 0, 1, 1, 0]`

### Two-Point Crossover

Two crossover points are selected, and the genetic material between these points is swapped between parents.

### Uniform Crossover

For each gene position, randomly choose which parent to inherit from (typically with 50% probability for each parent). This approach provides more mixing than point-based methods.

### Order Crossover (OX)

Specifically designed for permutation problems like the Travelling Salesman Problem, where each gene must appear exactly once. In this example, we can switch around nodes (if it generates a valid graph).

## [Mutation](mutation.md)

In the mutation step, we modify the child solution slightly, allowing us to maintain genetic diversity and continue exploring the solution space. There are many mutation algorithms.

### Bit-flip

For binary representations, each bit has a small probability (typically 1-5%) of being flipped from 0 to 1 or vice versa.

### Gaussian

For real-valued genes, add a random value drawn from a Gaussian (normal) distribution with mean zero and a small standard deviation.

Formula: $\text{new gene} = \text{old gene} + \mathcal{N}(0, \sigma)$

### Swap Mutation

For permutation problems, randomly select two positions and swap their values.

### Insertion Mutation

Remove a gene from one position and insert it at another random position.

## [Replacement](replacement.md)

Finally, we replace some or all of the old population with the new offspring.

## Advantages and Limitations

There are a few advantages to genetic algorithms over other AI approaches, including:
* No gradient information required.
* Can handle discrete, continuous and mixed variable types.
* Can typically be parallelised (evaluating the fitness function is the main computationally heavy part and can be done on multiple cores)
* Can escape local optima
* Robust to noise in fitness function

On the other hand, there's no guarantee of finding a global optimum; typically, many parameters need to be tuned, which can be computationally expensive and can easily lead to premature convergence.