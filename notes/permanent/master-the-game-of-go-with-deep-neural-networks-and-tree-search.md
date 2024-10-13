---
title: "Mastering the Game of Go with Deep Neural Networks and Tree Search"
date: 2024-10-10 00:00
modified: 2024-10-10 00:00
status: draft
---

Notes from paper [Mastering the game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961) by David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel & Demis Hassabis.

## Overview

This paper from DeepMind describes [AlphaGo](alphago.md), a program that uses deep neural networks to evaluate board positions and select moves, trained through a combination of [Supervised Learning](supervised-learning.md) and [Reinforcement Learning](../../../permanent/reinforcement-learning.md).

he program employs two neural networks:
* a value network that estimates the probability of winning from a given board position
* a policy network that predicts the best move to make.

These networks are trained using a combination of supervised learning, where they learn from human expert games, and reinforcement learning, where they improve their performance by playing against themselves.

AlphaGo’s uses [Monte Carlo Tree Search](../../../permanent/monte-carlo-tree-search.md) to explore possible game sequences by combining the information from the neural networks with traditional Monte-Carlo rollouts, creating a powerful and efficient search engine.
