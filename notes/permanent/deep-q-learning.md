---
title: "Deep-Q Learning"
date: 2025-03-29 00:00
modified: 2025-03-29 00:00
status: draft
---

**Deep-Q Learning** (DQN) is a variant of [Q-Learning](q-learning.md) sthat uses a deep neural network to approximate the Q-Values for each state.

**Q-Function Approximation**: Rather than maintaining unwieldy Q-value tables, DQN employs neural networks to approximate the Q-function, enabling generalisation across complex state spaces.

**Experience Replay**: The agent's experiences (state, action, reward, next state) are stored in a buffer and randomly sampled during training. Think of it as a student reviewing diverse past lessons rather than only the most recent ones, reducing correlation between consecutive updates and stabilising learning.

**Target Network**: A separate network computes target Q-values, updated less frequently than the main network. This prevents the "moving target" problem—imagine trying to hit a bullseye that constantly shifts position.

**Loss Function**: DQN minimises the difference between predicted and target Q-values:

$L(\theta) = \sum_{(s,a,r,s')\in D}[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]$

This temporal difference error drives the network toward better predictions.

**Epsilon-Greedy Exploration**: Starting with low exploration ($\epsilon \approx 0.05$) and gradually increasing it allows the agent to initially rely on its Q-function while maintaining exploration flexibility. This contrasts with traditional approaches where epsilon typically decreases over time.

**State Transition Matrix**: In realistic environments, actions can lead to multiple possible next states, each with its own probability. DQN's architecture handles these stochastic transitions effectively, making it powerful for real-world applications with uncertain outcomes.