---
title: Lexical Sentiment Analysis
date: 2026-01-06 00:00
modified: 2026-01-06 00:00
status: draft
---

**Lexical Sentiment Analysis** is an alternative to supervised learning approach, and infers meaning from word-sentiment associations rather than learned statistical models.

Example from TextBlob:

```python
from textblob import TextBlob

def get_blob_sentiment(sentence):
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    print(sentence, "polarity is", polarity)
    return polarity
```

