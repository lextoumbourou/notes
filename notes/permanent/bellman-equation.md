---
title: Bellman Equation
date: 2025-04-15 00:00
modified: 2025-04-15 00:00
status: draft
---

The **Bellman equation** provides a recursive definition for the value of a state in a [Markov Decision Process (MDP)](markov-decision-process.md). It expresses the value of a state as the expected return of taking an action from that state and then following a particular policy. This recursive formulation is central to many reinforcement learning algorithms, including Q-learning and Value Iteration.

$$
\textcolor{magenta}{V^\pi(s)} = \mathbb{E}_\pi \left[ \textcolor{red}{R_{t+1}} + \textcolor{orange}{\gamma} \cdot \textcolor{blue}{V^\pi(S_{t+1})} \mid S_t = s \right]
$$

Where:

- $\textcolor{magenta}{V^\pi(s)}$: The **value** of state $s$ under policy $\pi$. It represents the expected total reward the agent can accumulate starting from state $s$, following policy $\pi$.

- $\mathbb{E}_\pi[...]$: The **expectation** over all possible actions and resulting transitions, assuming the agent follows policy $\pi$.

- $\textcolor{red}{R_{t+1}}$: The **immediate reward** received after taking an action in state $s$ at time $t$.

- $\textcolor{orange}{\gamma}$: The **discount factor**, a number between 0 and 1, reduces the importance of future rewards. A lower $\gamma$ prioritizes immediate rewards more heavily.

- $\textcolor{blue}{V^\pi(S_{t+1})}$: The value of the **next state**, indicating how good it is to be in the state that follows from the current one, assuming policy $\pi$ continues to be followed.

In other words, it's telling us: "What's the expected return if we start in state $s$, take an action according to policy $\pi$, receive an immediate reward, and then continue following policy $\pi$ from whatever next state we end up in?"