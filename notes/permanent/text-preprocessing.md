---
title: Text Preprocessing
date: 2025-11-03 00:00
modified: 2025-11-03 00:00
status: draft
---

For classical NLP, there are a number of text preprocessing techniques that are worth knowing.

## [Tokenisation](../../../permanent/tokenisation.md)

Tokenisation is the process of splitting text (or audio - [Audio Tokenisation](../../../permanent/audio-tokenization.md), or images [Image Tokenisation](../../../permanent/image-tokenisation.md)) into discrete units called tokens.

Modern tokenisers, like the OpenAI's tiktoken, operate at the subword level, splitting text into small units called token, some of which are individual words, others parts of word, or even single character. In tiktoken's case, the algorithm is called [Byte Pair Encoding](Byte%20Pair%20Encoding.md), which operates across pairs of bytes.

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode("hello world aaaaaaaaaaaa")
print(tokens)
```
<!-- nb-output hash="72a7bf1c97c66ae2" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">[15339, 1917, 264, 70540, 33746]

</pre>
</div>
<!-- /nb-output -->

```python
for token_id in tokens:
    print(token_id, "->", encoding.decode([token_id]))
```
<!-- nb-output hash="ca0445962d8625aa" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">15339 -&gt; hello
1917 -&gt;  world
264 -&gt;  a
70540 -&gt; aaaaaaaa
33746 -&gt; aaa

</pre>
</div>
<!-- /nb-output -->

Subword tokenisers are not the only game in town for NLP.
### [Word Tokenisation](../../../permanent/word-tokenisation.md)

Word tokens, as you might expect, operate at the word limit. That means they're likely limited to a vocbulary, and any word outside of the voculablry might be replaced with a *unknown word*. 

```python

import nltk

tokens = nltk.word_tokenize("Hello world aaaaaaaaaaaa")
print(tokens)
```
<!-- nb-output hash="2a64233a27fd7d59" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">['Hello', 'world', 'aaaaaaaaaaaa']

</pre>
</div>
<!-- /nb-output -->

### [Sentence Segmentation](../../../permanent/sentence-segmentation.md)

Sentence Segmentation is a technique for tokenisation where we split a text into sentences, typically using punctuation as split tokens.

Here's an example from nltk using the `punkt_tab` 

```python
import nltk, pprint
nltk.download("punkt_tab")
```
<!-- nb-output hash="6e645744f83ebb62" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stderr">... ... ... ... ... ... ... [nltk_data] Downloading package punkt_tab to /Users/lex/nltk_data...
[nltk_data]   Package punkt_tab is already up-to-date!
</pre>
<pre class="nb-stream-stdout">True

</pre>
</div>
<!-- /nb-output -->

```python
text = """He never even had the makings of a varsity athlete. Your point being what, Junior? Oh, forget it, he's just breaking balls. All right, one thought I had in the interest of harmony, maybe there could be a power-sharing situation.

The Sopranos have two bosses."""
```
<!-- nb-output hash="c6d101c0ff1c7b24" format="html" -->

<!-- /nb-output -->

```python
sentences = nltk.tokenize.sent_tokenize(text)
pprint.pprint(sentences)
```
<!-- nb-output hash="9001e809232d353f" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">['He never even had the makings of a varsity athlete.',
 'Your point being what, Junior?',
 &quot;Oh, forget it, he's just breaking balls.&quot;,
 'All right, one thought I had in the interest of harmony, maybe there could '
 'be a power-sharing situation.',
 'The Sopranos have two bosses.']

</pre>
</div>
<!-- /nb-output -->

We can see that the sentence tokeniser has split sentences based on punctuation.

## [Text Normalisation](Text%20Normalisation.md)

Once a word has been tokenised, a number of other techniques exist for normalising tokens.

### [Stemming](stemming.md)

Stemming is the process of converting a token like driving into its root form, in a very basic way by simply chopping off the end: `driving` - > `drive`. This is useful when using basic word normalisation, as it prevents the model from having to learn a entirely different representation for a very similar word. However, in practice this lack of sophistocation can still mean that similar words end up represented differently.

```python
import nltk

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
print(stemmer.stem("driving"))
```
<!-- nb-output hash="5a56fad459aa7db2" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">drive

</pre>
</div>
<!-- /nb-output -->

### [Lemmatisation](../../../permanent/lemmatisation.md)

Lemmatisation is the process of replacing synatically similar words with the same token. It tends to be a much more sophistocated algorithmn than a stemming algorithm, and often uses surrounding context of a word to determine synoms and such.

Example: "passing" → lemma "pass" + ING (present participle)
Example: "were" → "to be" (past tense, second person plural)

In modern generative AI, both stemming and lemmatisation tend not to be used, in favour of a loss-less tokenisation algorithm like BPE, as they can throw away information that can be useful to a model. Although, on smaller NLP datasets trained from scratch, they still make a lot of sense.


```python
import nltk

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
print(lem.lemmatize("driving"))
```
<!-- nb-output hash="832d68f932854b65" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stderr">... ... ... ... ... ... ... [nltk_data] Downloading package wordnet to /Users/lex/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
</pre>
<pre class="nb-stream-stdout">driving

</pre>
</div>
<!-- /nb-output -->
