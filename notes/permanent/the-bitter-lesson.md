---
title: "The Bitter Lesson"
date: 2026-10-18 09:22
modified: 2026-10-18 09:22
status: hidden
summary: "Rich Sutton's 2019 essay arguing that general methods that scale with computation beat methods built on human knowledge."
tags:
- MachineLearning
- HarnessDesign
---

Rich Sutton's 2019 essay arguing that general methods that scale with computation beat methods built on human knowledge [@suttonBitterLesson2019].

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."

His examples come from computer chess, Go, speech recognition and computer vision. In each, researchers first built human knowledge into their systems. That helped in the short term but eventually plateaued, and the breakthrough came from scaling computation through search and learning.

It's a common argument for [Thin Harnesses](thin-harnesses.md): as models improve, hand-written harness logic should give way to the model. The [Context Language Models](context-language-models.md) paper says its findings echo it.
