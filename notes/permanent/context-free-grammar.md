---
title: Context-Free Grammar
date: 2024-01-09 00:00
modified: 2024-01-09 00:00
status: draft
tags:
- NaturalLanguageProcessing
---

A **Context-Free Grammar** is a set of rules that expressed how symbols of a language can be grouped together in valid ways.

For example:

```
S -> NP VP
NP -> Det N
VP -> V NP
```

Here we have rules that say:
* A sentence (`S`) consists of a **noun phrase** followed by a **verb phrase**.
* A noun phrase (`NP`) is made from a determiner and a noun.
* A very phrase (`VP`) is made from a verb and a noun phrase.

These non-terminals (`S`, `NP`, `VP`) are [Constituents](constituency.md) of the syntax.

Take the sentence "the black cat chased the mouse"

Using the CFG rules, we can derive:

```
S
├── NP
│   ├── Det -> the
│   └── N   -> cat
└── VP
    ├── V  -> chased
    └── NP
        ├── Det -> the
        └── N   -> mouse
```

In the formal definition, we think of it as a 4-tuple ($V$, $\Sigma$, $R$, $S$) where:
* Variables: a finite set of symbols denoted by V
* Terminals: a finite set of letters denoted by $\Sigma$; it is disjoint from V
* Rules: a finite set of mappings, denotes by R, with each rule being a variables and a string of variables and terminals.
* Start variable: a member of V, denoted by S. Usually the variable on the left-hand side of the top rule.

We can think of CFG as having two layers of rules:
* Structural rules (rules about phrase structure)
- Lexical Rules or the *lexicon* (rules are mapping categories to words)

The "Lexical Rules" map syntactic categories to words:

```
Det -> the | a
N -> cat | mouse
V -> chased | caught
```

These rules allow abstract categories like `N` or `V` to produce actual tokens.

Without them, the grammar could generate trees but not sentences.




