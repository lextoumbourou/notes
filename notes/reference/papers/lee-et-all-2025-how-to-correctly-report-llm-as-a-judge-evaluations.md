---
title: "Lee et al., (2025) - How to Correctly Report LLM-as-a-Judge Evaluations"
date: 2025-11-29 00:00
modified: 2025-11-29 00:00
category: reference/papers
paper_url: https://arxiv.org/abs/2511.21140
year: 2025
doi:  https://doi.org/10.48550/arXiv.2511.21140
tags:
- LLMEvals
- LargeLanguageModels
status: draft
---

*My notes for the paper How to Correctly Report LLM-as-a-Judge Evaluations* [@leeHowCorrectlyReport2025]

The paper describes a way to calculate a confidence interval for [LLM-Judge](LLM-Judge.md) evaluations. It works something like this: you have your eval set of input prompts, and a model under test, and a way to evaluate your outputs as true / false (this paper doesn't handle multiple criterions, but I guess you could do this for each output criterion).

Compute your outputs given your inputs, then compute judge labels for your inputs.

The naive judged accuracy is just the standard accuracy:

$\hat{p} = \frac{1}{n} \sum\limits_{i=1}^{n} \hat{z}_i$

For a subset of those input / output pairs, get humans to label them based on your crierion.

m0 = # examples where human says z=0
m1 = # examples where human says z=1

Calculate specitificy (accuracy on incorrect cases):

$\hat{q}_0 = \frac{ \text{\# } z=0 \text{ and } \hat{z} = 0}{m0}$

Calculate sensitivity (judge's accuracy on correct cases):

$\hat{q}_1 = \frac{ \text{\# } z=1 \text{ and } \hat{z} = 1}{m1}$

Compute the bias-adjusted accuracy:

$$
\hat{\theta} = \frac{\hat{p} + \hat{q}_0 - 1}{\hat{q}_0 + \hat{q}_1 - 1}
$$

Clip to 0 and 1, giving you the best estimate of true human-defined correctness rate of code generator.

Now, you can plug that into their formula for confidence interval:

```
def confidence_interval(p, q0, q1, n, m0, m1, alpha=0.05):
    """Compute the adjusted (1-alpha) confidence interval."""
    z = norm.ppf(1 - alpha/2)

    # Laplace adjustments
    p = (n * p + z**2 / 2) / (n + z**2)
    q0 = (m0 * q0 + 1) / (m0 + 2)
    q1 = (m1 * q1 + 1) / (m1 + 2)

    n = n + z**2
    m0 = m0 + 2
    m1 = m1 + 2

    th = (p + q0 - 1) / (q0 + q1 - 1)

    dth = 2 * z**2 * (-(1 - th) * q0 * (1 - q0)/m0 + th * q1 * (1 - q1)/m1)

    se = sqrt(
        p*(1-p)/n +
        (1-th)**2 * q0*(1-q0)/m0 +
        th**2 * q1*(1-q1)/m1
    ) / (q0 + q1 - 1)

    return clip(th + dth - z*se), clip(th + dth + z*se)
```

