---
title: "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"
date: 2024-10-11 00:00
modified: 2026-09-26 08:55
status: draft
---

**HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering** is a 2018 paper by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov and Christopher D. Manning, published at EMNLP 2018 ([arXiv](https://arxiv.org/abs/1809.09600)).

It introduces [HotpotQA](hotpotqa.md), a question-answering dataset of around 113k question-answer pairs based on Wikipedia. Answering each question requires finding and reasoning over more than one supporting document (multi-hop reasoning).

The dataset also labels the sentence-level supporting facts needed to answer each question, so models can be trained and evaluated on explaining their answers, not just getting them right.
