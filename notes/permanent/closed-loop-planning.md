---
title: Closed-loop Planning
date: 2024-08-27 00:00
modified: 2024-08-27 00:00
status: draft
---

**Closed-loop planning** is an [[Reflection](../../../permanent/reflection.md)](agentic-reasoning.md) technique, where the model decides a next action based on the current state of the environment after executing the previous step. As opposed to [[../../../permanent/open-loop-planning]] where the entire action sequence is planned before execution.

Closed-loop simplifies planning, as the model doesn't have to reason through the complicated state changes, and also allows the system to adapt to changes in the environment and improve its performance over time.

Features a loop of continuous planning, execute and feedback.

* 1. Get next step.
* 2. Validate and run. 
* 3. Update plan.
* Repeat.