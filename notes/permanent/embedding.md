---
title: Embedding
date: 2022-03-04 00:00
modified: 2026-05-25 00:00
summary: A learned vector representation that encodes meaning as a point in space, where similar things end up close together.
tags:
- MachineLearning
- LargeLanguageModels
category: note
alias:
- Embeddings
---

An **embedding** is a [Vector](vector.md) of numbers that represents something - a word, a token, an image, a user - in a way that captures meaningful relationships. Things that are semantically similar end up close together in that space.

## The intuition: recommending movies

Imagine building a movie recommender. One approach is to create a spreadsheet where each column is a feature like "how romantic is it?" or "how much action?", and have users rate each feature. Then you could plot movies and see clusters of similar ones. Die Hard and Speed end up near each other. You could say: if you liked one, you'll probably like the other.

You could make a similar spreadsheet for each user, rating how much they care about each feature, then find movies that best match their profile.

The problem is that coming up with the right features by hand relies entirely on your intuition about what matters to users. You will miss things. And asking users to fill in feature ratings for every movie they watch definitely won't scale.

What if instead you showed the model many examples of what movies people actually liked, and let it figure out the useful features on its own? You train it on a large dataset of user ratings, asking it to predict how much a given user will enjoy a given movie. To do that well, the model learns a vector for each movie and each user. Nobody tells it what the dimensions mean, but after training, movies cluster into meaningful groups. Dimensions loosely correspond to things like genre, tone, or era.

This is the key insight of embeddings: useful structure emerges from learning to predict.

## Token embeddings

In language models, the input is a sequence of tokens (roughly, words or subwords). Each token gets assigned an embedding - a vector of typically hundreds or thousands of numbers - via a learned lookup table called `nn.Embedding`.

```python
import torch.nn as nn

vocab_size = 10_000
embedding_dim = 512

embed = nn.Embedding(vocab_size, embedding_dim)
```

When you pass a token index, you get back its embedding vector:

```python
token_id = torch.tensor([42])
vec = embed(token_id)  # shape: (1, 512)
```

The weights of this lookup table are learned alongside the rest of the network. After training on enough text, similar words end up with similar vectors. The classic demonstration is:

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

The arithmetic works because the model has learned that the "royalty" direction and the "gender" direction are separable in the embedding space.

## The limitation

Token embeddings are **static**. Every occurrence of the word "minute" gets the same vector, regardless of whether it refers to time or size. The embedding captures what a word generally means, but not what it means in context.

This is the problem that the [Attention Mechanism](attention-mechanism.md) was designed to solve: given a sequence of token embeddings, compute a new representation for each token that incorporates information from the surrounding tokens.
