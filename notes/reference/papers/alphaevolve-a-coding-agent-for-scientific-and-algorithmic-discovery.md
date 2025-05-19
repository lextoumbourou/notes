---
title: "AlphaEvolve: A coding agent for scientific and algorithmic discovery"
date: 2025-05-18 00:00
modified: 2025-05-18 00:00
summary: "vibe-coding new math discoveries"
category: reference/papers
tags:
- AgenticReasoning
- LargeLanguageModels
- EvolutionaryAlgorithms
---

*My (WIP) summary of the white paper from DeepMind: [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) by Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli and Matej Balog*

You may have heard of DeepMind's new whitepaper, AlphaEvolve. It discovered a new algorithm for multiplying matrices, improving on a 56-year-old solution, and improving many other mathematical problems, including improving Google's internal infrastructure scheduling algorithms.

The solution is an evolution of DeepMind's 2023 work called FunSearch, where the basic idea is to continuously refine the LLM-generated code solution (prompting in a while loop) based on evaluation metrics using an evolutionary sampling algorithm:

1. Start with a prompt describing a problem, including a code base skeleton, with **EVOLVE-BLOCK** markers to show where the LLM should modify the code.
2. Use an evaluation function that assesses the solution's correctness and measures other properties, such as runtime, simplicity, etc.
3. Create a program database, which can start empty or be seeded with known solutions.
4. Sample programs from the database, using an evolutionary selection strategy (see below).
5. Instruct a set of LLMs to generate new programs to improve the evaluation metrics.
6. Now evaluate the results, and store the most promising programs.
 7. Repeat until you have SOTA.

The sampling algorithm is based on prior work on evolutionary algorithms (MAP-Elites and island models):

- Solutions are clustered into "islands" based on performance characteristics.
* Sample high-scoring programs from an island as examples (selection)
* LLM generates new programs by combining programs to create new variations (breeding/variation)
* Weak islands are periodically culled and repopulated with some successful approaches.

The main improvements on FunSearch are that a) it can evolve an entire codebase, not just a single function, b) they have access to more powerful LLMs, from PaLM2 to Gemini Flash/Pro 2.0. Presumably, even more improvements can be made by utilising the latest generation of Gemini models (2.5 Pro, etc).

One interesting detail is that the open problems were suggested by mathematician Javier Gomez Serrano and Terence Tao, who also helped "formulate them as inputs". So in a way, this might be the first example of mathematical discoveries made from vibe-coding.

![alphaevolve-fig-2.png](../../_media/alphaevolve-fig-2.png)