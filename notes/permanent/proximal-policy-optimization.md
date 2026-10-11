---
title: Proximal Policy Optimization
date: 2025-01-31 00:00
modified: 2026-09-26 08:55
status: draft
---

**Proximal Policy Optimization** (PPO) is a policy gradient algorithm for [Reinforcement Learning (RL)](reinforcement-learning.md), introduced by John Schulman and others at OpenAI in the 2017 paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

![Title of the paper Proximal Policy Optimization Algorithms by John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov of OpenAI](../_media/proximal-policy-optimization-title.png)

It improves the policy in small steps. Its clipped objective stops the new policy from moving too far from the old one in a single update:

$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t\right)\right]$

where $r_t(\theta)$ is the ratio between the probability of the action under the new and old policies, and $\hat{A}_t$ is the estimated advantage.

This makes training more stable than vanilla policy gradient methods, while being simpler to implement than TRPO. PPO became one of the most widely used RL algorithms, including for [Reinforcement Learning from Human Feedback](reinforcement-learning-from-human-feedback.md) in language models. See also [Group Relative Policy Optimisation](group-relative-policy-optimisation.md).

![Figure 1 from the PPO paper: two plots of one term of the clipped objective L^CLIP against the probability ratio r. For a positive advantage, the objective rises with r and flattens beyond 1 + epsilon. For a negative advantage, it is flat below 1 - epsilon and falls as r increases. A red dot marks the starting point r = 1 on each plot](../_media/proximal-policy-optimization-fig-1.png)

![Figure 3 from the PPO paper: learning curves over one million timesteps on seven MuJoCo environments (HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer and Walker2d), comparing A2C, A2C with trust region, CEM, PPO (Clip), adaptive vanilla policy gradient and TRPO. PPO reaches the highest or near-highest reward in most environments](../_media/proximal-policy-optimization-fig-3.png)

![Table 2 from the PPO paper: number of Atari games won by each algorithm, scores averaged over three trials. By average episode reward over all of training, A2C wins 1, ACER 18, PPO 30 and 0 are tied. By average episode reward over the last 100 episodes, A2C wins 1, ACER 28, PPO 19 and 1 is tied](../_media/proximal-policy-optimization-table-2.png)
