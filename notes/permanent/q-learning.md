---
title: "Q-Learning"
date: 2025-04-18 00:00
modified: 2025-04-21 00:00
summary: an algorithm for finding optimal policies in a Markov Decision Process model
tags:
- ReinforcementLearning
cover: /_media/taxi-q-learning-example.png
---

This article is part of my (WIP) series on [Reinforcement Learning](reinforcement-learning.md).

**Q-Learning** is a reinforcement learning algorithm for finding optimal policies in [Markov Decision Process](markov-decision-process.md). It can learn the value of actions without requiring a model of the environment, hence it's called a "model-free" method.

The algorithm was introduced by Chris Watkins in his 1989 PhD thesis at King's College [^1], with a convergence proof later published by Watkins alongside Peter Dayan [^2]. The concept was also developed independently by several others in the early 90s.

## How It Works

Q-Learning is an **iterative optimisation algorithm**, like [Gradient Descent](gradient-descent.md). It refines value estimates based on interaction with the environment, utilising the **Bellman Equation** to update its predictions iteratively.

At the core of Q-Learning is a lookup table called a **Q-Table**, which maps state-action pairs to expected future rewards. The values in the **Q-Table** are called **Q-Values**.

Rewards are discounted, so immediate gains matter more than distant ones.

The algorithm uses three key hyperparameters:

- **Learning Rate** $\alpha$ – how quickly the algorithm updates its estimates.
- **Discount Factor** $\gamma$ – how much it values future rewards versus immediate ones.
- **Exploration Rate** $\epsilon$ – how often it chooses a random action over the current best action.

The $\epsilon$ parameter balances the trade-off between **exploration and exploitation**, managing how much the agent tries new things versus sticking to what it already thinks is best.

## Q-Learning in Practice: Taxi

We'll use the `Taxi-v3` environment from the `gymnasium` library. In this environment, the agent (a taxi) must navigate a grid to pick up and drop off passengers at the right location.

### Setup Code

You can install Gymnasium with the toy-text dependencies

```bash
pip install "gymnasium[toy-text]"
```

Then run this code in a Python script:

```python
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from pathlib import Path
from tqdm import tqdm

# Q‑learning hyperparameters
alpha, gamma, epsilon = 0.1, 0.99, 0.1

# An episode is a start to finish exploration of the state space,
# where the finish is either reaching a goal or a hazard.
episodes = 10_000

env = gym.make("Taxi-v3", render_mode="human")

n_states, n_actions = env.observation_space.n, env.action_space.n
Q = np.zeros((n_states, n_actions))

for ep in tqdm(range(episodes), desc="Training episodes"):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q‑learning update using Bellman Equation
        td = reward + gamma * np.max(Q[next_state]) - Q[state, action]
        Q[state, action] += alpha * td
        state = next_state

env.close()
```

You should see a very slow visual representation of the agent exploring and slowly updating its policy.

![taxi_q_learning-episode-0](_media/taxi_q_learning-episode-0.mp4)

If you had the patient to wait for that to finish, eventually you would see the finish episode, on a fully trained policy.

![taxi_q_learning-episode-0](_media/taxi_q_learning-episode-9999.mp4)

As you can see, the taxi can pick the passenger up and drop them off at the destination in a very direct way.

With a trained policy, we can use `argmax` to pick the highest reward state at every step:

```python
state, _ = env.reset()
env.render()
done = False

while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print(env.render())
```

Now, let's take a closer look at the update rule.
## Q-Learning Update Rule

The core of the Q-Learning algorithm is its update rule, which is derived from the [Bellman Equation](bellman-equation.md):

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \Big]
$$

Where:

- $Q(s,a)$: Current value of taking action $a$ in state $s$
- $\alpha$: Learning rate
- $r$: Reward received after taking the action
- $\gamma$: Discount factor
- $\max_{a'} Q(s', a')$: Best estimated future value

It measures how far off the current estimate is from the ideal update, nudging Q-values toward better predictions.

## Deep-Q Learning

In 2013, DeepMind published a paper where they replaced the Q-table with a [Neural Network](../../../permanent/neural-network.md) to approximate the Q-values — a technique known as [deep-q-learning](../../../permanent/deep-q-learning.md)[^3].

[^1]: Watkins, C. J. C. H. (1989). *Learning from delayed rewards* (Doctoral dissertation). King's College, Cambridge.
[^2]: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3-4), 279–292. https://doi.org/10.1007/BF00992698
[^3]: Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2013). Playing Atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*. https://arxiv.org/abs/1312.5602