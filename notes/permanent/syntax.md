---
title: Syntax
date: 2026-01-08 00:00
modified: 2026-01-08 00:00
status: draft
tags:
- NaturalLanguageProcessing
- Linguistics
---

**Syntax** is a high-level concept in NLP refers to how words are arranged and combined into valid sentences.

Words belong to categories (noun, verb, determiner, etc), and these categories can be combined in constrained, rule-governed ways.

Core of the core syntactical ideas is the idea of [Constituents](constituency.md)
## Constituency

A group of words behaves as a single unit; its constituents can be substituted into similar syntactic environments.

Consider the sentence:

> "The black cat chased the mouse"

A syntactic claim is that a group of words behaves as a single unit (constituents), not just individual words.

In this example:
* The black cat
* the mouse

These groups act as [Noun Phrases](noun-phrases.md) (NPs). We can show this using **constituency tests**.

1. Substitution Test

If a group of words can be replaced by a single word (like a pronoun) without breaking the sentence, it is likely a constituent.

> She chased the mouse.
> The black cat chased him.

Since both substitutions are grammatical, it suggests that *the black cat* and *the mouse* are constituents.

2. Movement tests

Constituents can be moved together.

> The mouse, the black cat chased.

Sounds like something Yoda would say, but it's grammatically valid. Again, it tells us "the mouse" is a single syntactic unit.

3. Coordination Test

Only constituents of the same type can be coordinated with other constituents of the same kind.

> The black cat and a big dog chased the mouse.

"The black cat" and "a big dog" coordinate naturally together, reinforcing that it's a noun phrase.

The concept of Constituency is central to syntactic parsing. Parsers don't just label words with parts of speech; they build hierarchical structures, identifying phrases and how they nest inside one another.

A [Context-Free Grammar](context-free-grammar.md) is one of the main ways to represent syntactic structure in NLP. CFG defines rules that build constituents from smaller constituents. There are also [Dependency Grammar](dependency-grammar.md)

We can think of constituency as operating at the "phrase-level", grouping words into units like:
- Noun Phrase (NP) - "the black cat"
- Verb Phrase (VP) - "chased the mouse"
- Prepositional Phrase (PP) - "in the playroom"
- Adjective Phrase (AP) - very tired

When we combine "phrase-level constituencts" into sentences, we have a new set of categories called "sentence-level constructions".

## [Sentence-Level Constructions](Sentence-Level%20Constructions.md)

These are sets of phrase-level constituents that combine together to create a new set of categories. Some of the categories include:

### 1. Declarative Sentences

Statements of fact or belief.

```
S -> NP VP
```

> The cat chased the mouse.

This is the canoncical sentence form in English and the backbone of most grammars.

### 2. Yes - No Questions Sentences

These are sentences formed via auxillary inversion.

```
S -> Aux NP VP
```

> Did the dog chase the ball?
> Is she working?

The key property is that they have subject-auxilary inversion.

### 3. Wh-Questions Sentences

Questions that extract a constituent.

```
S -> Wh-Phrase Aux NP VP
```

> What did the dog chase?
> Who is coming?

These combine:
* Movement (wh-fronting)
* Gaps / traces (implicit missing constituents)

### 4. Imperative Sentences

Commands or requests.

```
S -> VP
```

> Close the door
> Chase the ball.

The subject ("you") is implicit, and not syntactically present.

### 5. Passive Sentences

Sentences where the object is promoted to subject position.

```
S -> NP Aux VP[passive]
```

> The mouse was chased (by the cat).

Still an `S -> NP VP` structure, but with:
- Passive morphology.
- Optional agent phrase.

### 6. Copular Sentences

Linking a subject to a predicate.

S -> NP Copular XP

Where `XP` can be:

* NP (She is a doctor)
* AP (She is tired)
* PP (She is in the room)

### 7. Existenial Constructions

Introduce the existence of something

```
S -> There Aux NP
```

> These is a problem
> There are two solutions

Note: "there" is expletivie ,not referential.

### Embedded / Clause-Level Sentences

These are still sentence constructions, but not root sentences.

### 8. Complement Clauses

Sentences embedded inside other sentences.

```
VP -> V CP
CP -> (that) S
```

> `I think [that she left]`.

### 9. Relative Clauses

Modify noun phrases.

```
NP → NP CP
```


> `The dog [that chased the ball]`

### 10. Subordinate Clauses

Adverbial sentence modifiers.

```
S → Subordinator S , S
```

> Because it was raining, we stayed inside.