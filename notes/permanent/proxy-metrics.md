---
title: Proxy Metrics
date: 2021-06-12 00:00
tags:
  - MachineLearningFailureModes
summary: Metrics are usually a proxy for what we really care about
cover: /_media/boat-equipment.jpg
---

Some of the time, it is difficult to measure exactly what we need to assess a problem objectively. Instead, we settle for metrics that are "a proxy for what we really care about".

[@thomasProblemMetricsFundamental2020].

For example, well-meaning technology decision-makers realise that unit tests are important for sustainable growth of software projects. Therefore, to ensure that tests are used liberally they introduce [Test Coverage Metrics](test-coverage-metrics.md). And sometimes even go as far as failing to promote builds where coverage metrics have fallen below some threshold. However, test coverage metrics can be gamed in 2 ways:

* code be reshuffled to reduce the number of lines and therefore increase % of lines covered by the test. This reshuffling can sometimes make the code less readable
* tests can be written to assert nothing of value to simply increase branch or line coverage. This can unneccessarily increase complexity of the code.

These outcomes are the opposite of the initial goals.

Test quality is much more important than test coverage and cannot be measure by metrics.

[@khorikovUnitTestingPrinciples2020]

This phenomenon is particularly a problem in Machine Learning, where metric optimisation is foundational.

In Kaggle competitions, tricks that optimise the metrics in a way that doesn't align with the organiser's goals is called [Metric hacking](Metric hacking).

Cover image by [Mikail McVerry](https://unsplash.com/photos/-yBvef_mAaQ)
