---
title: On the Measure of Intelligence (2019)
date: 2024-10-22 00:00
modified: 2024-10-22 00:00
summary: a shift in how we evaluate artificatial intelligence
cover: _media/on-the-measure-of-intelligence-2019-fig3.png
status: draft
---

*My notes from paper [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547) (2019) by François Chollet.*

---

## Overview

To make progress towards AGI, we need better ways to define and evaluate intelligence.

AI is typically measured by comparing performance at specific tasks, like games. However, solely measuring skills at a task doesn't measure intelligence, because the capability in a skill is determined by prior knowledge. With unlimited priors or unlimited training data allow experimenters to “buy” arbitrary levels of skills for a system, in a way that masks the system’s own generalisation power.

In this paper, a new formal definition of intelligence based on [[../../../../permanent/algorithm Information Theory]] is described where they describe intelligence based on [Skill-Acquisition Efficiency](../../../../permanent/Skill-acquisition%20Efficiency.md). They show that generalisation difficulty, priors, and experience, as
critical pieces to be accounted for in characterising intelligent systems, and they create guidelines for general AI benchmarks.

With these guidelines, they propose a new benchmark [Abstraction and Reasoning Corpus](../../../../permanent/abstraction-and-reasoning-corpus.md) which is "built upon an explicit set of priors designed to be as close as possible to innate human priors", and can be used to measure a "human-like form of general fluid intelligence", which gives more fair comparisons between AI systems and humans.

## 1.1 Need for "actionable" definition

* AI's original promise was to create human-like intelligence, but so-far only created narrow, task-specific systems.
* The field critically needs precise, quantitative definitions and measurements of intelligence that are **actionable**, **explanatory**, and **measurable**

## 1.2 Defining intelligence: two divergent visions

> In the context of AI research, Legg and Hutter [53] summarized in 2007 no fewer than 70 definitions from the literature into a single statement: “Intelligence measures an agent’s ability to achieve goals in a wide range of environments.”

Two characterisations:
* one with an emphasis on task-specific skill (“achieving goals”)
* one focused on generality and adaptation (“in a wide range of environments”)

### 1.2.1 Task-specific

* Rooted in evolutionary psychology
* Sees intelligence as collection of specialized adaptations
* Championed by Marvin Minsky
* Led to focus on programming specific tasks rather than learning
* Resulted in paradox: AI systems performing tasks without true intelligence

### 1.2.2 General Learning Ability

* Views intelligence as flexible ability to learn new skills
* Championed by Turing, McCarthy, and others
* Emphasized machines learning without explicit programming
* Connected to Locke's "Tabula Rasa" (blank slate) theory
* Gained prominence with rise of machine learning and deep learning
* Both views are likely **incorrect**.

## I.3 AI evaluation: from measuring skills to measuring broad abilities

* Skill-based narrow AI

### The Spectrum of Generalization

Chollet describes a spectrum of generalisation capabilities:

* Absence of generalisation - i.e. a game that plays tic-tac-toe via an exhaustive iteration of all possibilities)
* Local generalisation (robustness) - an image classifier that can distinguish previously unseen 150x150 RGB images containing cats from those containing dogs, after being trained on many such labelled images, can be said to perform local generalisation
* Broad generalisation (flexibility) - i.e. L5 self-driving vehicle or a domestic robot capable
of passing Wozniak's coffee cup test (entering a random kitchen and making a cup of
coffee) [99] could be said to display broad generalisation
* Extreme generalisation (general intelligence) - i.e. biological forms of intelligence (humans and possibly other intelligent species) are the only example of such a system at this time

![](../../../../_media/on-the-measure-of-intelligence-2019-fig-1.png)
*Figure 1. Hierarchical model of cognitive abilities and its mapping to the spectrum of generalisation.*

This spectrum aligns with the hierarchical structure of cognitive abilities in psychometrics, with general intelligence (the 'g factor') at the apex.

### Formalizing Intelligence Measurement

Using [[Algorithmic Information Theory]], Chollet provides a formal definition of intelligence that quantifies:

* Generalisation difficulty
* Prior knowledge
* Experience

This formalisation allows for more rigorous comparisons between AI systems and even between AI and human intelligence.

![](../../../../_media/on-the-measure-of-intelligence-2019-fig3.png)
*Figure 3: Higher intelligence "covers more ground" in future situation space using the same information*

### Abstraction and Reasoning Corpus (ARC)

To implement his ideas, Chollet introduces the [[Abstraction and Reasoning Corpus]] (ARC), a benchmark designed to evaluate general intelligence in both AI systems and humans. Key features of ARC include:

* Abstract visual reasoning tasks
* Novel problems in the evaluation set
* Explicit [[Core Knowledge]] priors
* Limited training examples

![](../../../../_media/on-the-measure-of-intelligence-2019-fig4.png)
*Figure 4: A task where the implicit goal is to complete a symmetrical pattern. Three input/output examples specify the nature of the
task. The test-taker must generate the output grid corresponding to the input grid of the test input (bottom right).*

ARC aims to measure developer-aware generalisation, ensuring high scores reflect genuine problem-solving abilities rather than memorisation or exploiting task-specific knowledge.

### Implications for AI Research and Development

Chollet's framework has several important implications:

* Emphasis on program synthesis over task-specific training
* Incorporation of human-like cognitive priors
* Focus on curriculum optimisation for efficient learning
* Prioritisation of information efficiency in learning processes
* Development of strong abstraction capabilities for extreme generalisation

### Challenges and Limitations

While promising, this new approach faces several challenges:

* Difficulty in quantifying prior knowledge
* Complexity in designing truly novel tasks for evaluation
* Potential resistance from researchers invested in current benchmarks
* Computational challenges in implementing ARC-like evaluations at scale
