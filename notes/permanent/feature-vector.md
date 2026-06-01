---
title: Feature Vector
date: 2026-05-26 00:00
modified: 2026-05-26 00:00
status: draft
aliases:
- Feature Vectors
---

A **feature vector** is a [Vector](vector.md) of numbers that represent some object.

Picture a dog adoption centre that helps you find a dog breed based on your preferences and circumstances.

They might first create a spreadsheet with each dog breed and a rating for each of its unique characteristics, for example, how active it is, how big it grows to full size, and how good it is around kids.

```python
import pandas as pd

dogs = pd.DataFrame({
    'breed':        ['Labrador', 'Chihuahua', 'Border Collie', 'Basset Hound', 'Golden Retriever'],
    'activity':     [4,          3,            5,               2,              4],
    'size':         [4,          1,            3,               3,              4],
    'kid_friendly': [5,          2,            3,               4,              5],
})

dogs
```
<!-- nb-output hash="3382718472e552dd" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><div>
<style>.dataframe tbody tr th:only-of-type { vertical-align: middle; } .dataframe tbody tr th { vertical-align: top; } .dataframe thead th { text-align: right; }</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>breed</th>
      <th>activity</th>
      <th>size</th>
      <th>kid_friendly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Labrador</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chihuahua</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Border Collie</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basset Hound</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Golden Retriever</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div></div>
</div>
<!-- /nb-output -->

If we have our resident breed expert fill out the spreadsheet for us, with a rating between 1 and 5 for each feature, we now have a **feature vector** for each dog, based on the features we defined.

How can we use this?

Now, what if we gave each adoptee a survey where they could describe their preferences for dogs using the same features:
* How active are you? (1–5)
* How much space do you have? (1–5)
* Do you have, or intend to have, young kids? (1–5)

```python
adoptees = pd.DataFrame({
    'name':         ['Alice', 'Bob', 'Carol'],
    'activity':     [4,       2,     5],
    'size':         [3,       2,     4],
    'kid_friendly': [5,       3,     2],
})

adoptees
```
<!-- nb-output hash="92582d00872630f6" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><div>
<style>.dataframe tbody tr th:only-of-type { vertical-align: middle; } .dataframe tbody tr th { vertical-align: top; } .dataframe thead th { text-align: right; }</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>activity</th>
      <th>size</th>
      <th>kid_friendly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alice</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carol</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div></div>
</div>
<!-- /nb-output -->

Now we have a feature vector with the same feature values as the dogs, and we can use this to compute the distance between vectors.

We can plot both in the same space to see which breeds land closest to each adoptee:

![[../../../_media/feature-vector-scatter.png]]

We could simply take each feature from the dog vector, and subtract it from the person preference vector, and remove the negative term (aka the [Absolute Value](../../../from-evernote/Maths/Absolute%20Value.md)) and find the dog with the smaller distance. That distance measure is called [Manhattan Distance](../../../permanent/manhattan-distance.md).

```python
dogs_indexed = dogs.set_index('breed')
adoptees_indexed = adoptees.set_index('name')

distances = pd.DataFrame(
    {name: (dogs_indexed - row).abs().sum(axis=1) for name, row in adoptees_indexed.iterrows()},
    index=dogs_indexed.index
)

print(distances)
print()
print("Best match per adoptee:")
print(distances.idxmin())
```
<!-- nb-output hash="fec068e906176cd6" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">                  Alice  Bob  Carol
breed                              
Labrador              1    6      4
Chihuahua             6    3      5
Border Collie         3    4      2
Basset Hound          3    2      6
Golden Retriever      1    6      4

Best match per adoptee:
Alice         Labrador
Bob       Basset Hound
Carol    Border Collie
dtype: object

</pre>
</div>
<!-- /nb-output -->

Or we could use the [Dot Product](dot-product.md) to measure the angle between two vectors.

Now, that's all well and good, but are sure we've picked all the right features? What about if the amount the dog sheds turns out to be very important? Or maybe people want a relatively quiet dog? Or maybe we prefer ball-driven dogs? There could be a lot of different features.

Also, maybe our users find the questions hard to answer. Instead they just tell you some dogs they have previously liked, and asked for a similar breed.

In other words, can we learn feature vectors based on users past behaviour? 

That's where an [Embedding](embedding.md) comes in.