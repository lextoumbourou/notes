---
title: TF-IDF
date: 2023-04-09 00:00
modified: 2025-12-20 00:00
summary: A word vectorisation technique that weights terms by their importance to a document relative to a corpus.
cover: /_media/tf-idf-cover.png
tags:
- NaturalLanguageProcessing
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

**TF-IDF** (Term Frequency - Inverse Document Frequency) is a numerical statistic used in Natural Language Processing to reflect how important a word is to a document in a collection or corpus.

The fundamental intuition is that:

1. Words that appear frequently in a specific document should be weighted higher.
2. Words that appear frequently across all documents (like "the", "is", and "and") have less signal and are weighted lower.

#### How it works:

**1. Term Frequency (TF)**

Measures how frequently a term occurs in a document.

$$\text{tf}(t, d) = \frac{\text{count of \textit{t} in \textit{d}}}{\text{total words in \textit{d}}}$$

**2. Inverse Document Frequency (IDF)**

$$\text{idf}(t) = \log\left(\frac{N}{1 + df_t}\right)$$

Measures how rare a term is across the entire corpus of documents.

* $N$: Total number of documents.
* $df_t$: Number of documents containing the term.
* The `+1` (smoothing) prevents division by zero if a term isn't in the training corpus.
* The `log` function dampens the magnitude of the IDF weight, ensuring that extremely rare words don't overpower the entire vector.

**3. TF-IDF Score**

The final score is the product of these two metrics:

$$\text{tf-idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$

Words that are frequent in a specific document but rare across the corpus receive the highest scores, making them the "signature" terms for that document.

---

We can visualise TF-IDF using a quick heatmap of 4 short documents with a limited vocabulary.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets",
    "the mat is on the floor"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=vectorizer.get_feature_names_out(),
    index=[f"Doc {i+1}" for i in range(len(corpus))]
)

plt.figure(figsize=(10, 6))
sns.heatmap(df_tfidf, annot=True, cmap="YlGnBu")
plt.title("TF-IDF Scores Heatmap")
plt.xticks(rotation=45)
plt.show()
```
