---
title: Mixture Model
date: 2024-04-11 00:00
modified: 2026-09-26 08:55
status: draft
---

A **Mixture Model** is a probabilistic model that represents data as a combination of several component distributions, where each observation is assumed to come from one of the components.

A **Mixture Model** allows for soft assignment of observations to clusters, unlike [K-Means](k-means.md): each observation gets a probability of belonging to each cluster, rather than a single hard label.

The most common example is the [Gaussian Mixture Model](gaussian-mixture-model.md), which is usually fitted using the Expectation-Maximisation (EM) algorithm.
