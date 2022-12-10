---
title: Scale Invariant Fine-Tuning
---

Scale Invariant Fine-Tuning is an [[Adversarial Training]] techique introduced in the [Deberta](https://arxiv.org/pdf/2006.03654.pdf) paper from Deberta, which adds some random information (or "pertubations") to the embedding vectors.

The aim is to force the model to perform well on so called "adversarial" examples in the aim to make it generalise better.

The implementation provided by [Deberta](https://github.com/microsoft/DeBERTa) turns out to be very easy to add into your pipeline:

```
# Create DeBERTa model
adv_modules = hook_sift_layer(model, hidden_size=768)
adv = AdversarialLearner(model, adv_modules)

def logits_fn(model, *wargs, **kwargs):
    logits,_ = model(*wargs, **kwargs)
    return logits

logits, loss = model(**data)

loss = loss + adv.loss(logits, logits_fn, **data)
# Other steps is the same as general training.
```

It was used in part of the 5th place solution described in @housuke's [post](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347379)

"I set learning_rate=0.2 and started giving perturbations after the CV score fell below 0.65."

I've modified the HuggingFace training to include SiFt in [this] notebook.

In my implementation, I start adversarial training on the 2nd epoch.

You can see from the comparison table that it makes the model perform a little better but nothing amazing. Still, every little bit helps