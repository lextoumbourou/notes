---
title: "LLMs Corrupt Your Documents When You Delegate"
date: 2026-05-09 00:00
modified: 2026-05-09 00:00
category: reference/papers
cover: /_media/llms-corrupt-cover.png
hide_cover_in_article: true
summary: "A large-scale study on long-horizon document tasks."
bluesky_post: https://bsky.app/profile/notesbylex.com/post/3mlf6gyxguc2r
mastodon_post: https://fedi.notesbylex.com/@lex/116542213243149508
linkedin_post: https://www.linkedin.com/feed/update/urn:li:activity:7458690168427003904
tags:
- AgenticReasoning
- LimitationsofLLMs
---

*My notes on [LLMs Corrupt Your Documents When You Delegate](https://arxiv.org/pdf/2604.15597) by Philippe Laban, Tobias Schnabel and Jennifer Neville from Microsoft Research.*

An interesting paper from researchers at Microsoft.

They introduce a benchmark called [DELEGATE-52](../../permanent/delegate-52.md), which contains a series of tasks across 52 domains that simulate long-delegate workflows for document editing, things like splitting a CSV into separate files based on categories, transposing the key of a music score, and manipulating graph diagrams. Every set of instructions in the benchmark is lossless and reversible, allowing them to measure how much each task degrades the information in the file when the task is reversed.

The authors demonstrated that in almost all cases, the information in the document would degrade over time. Even with the latest frontier models, users would see an average of 25% document corruption by the end of a long workflow, with all tested models averaging 50% corruption [@labanLLMsCorruptYour2026].

The only domain that was nearly completely safe from degradation (considered "delegation ready") was Python.

![Three examples of document degradation over 20 interactions: a Linux Kernel Architecture graph diagram losing nodes and edges, a 12-Shaft Twill Diamond textile pattern becoming corrupted, and an ActionBoy Palm Tree 3D object losing geometry. Each shows progressive corruption from interaction 4 to 20.](../../_media/llms-corrupt-figure-1-examples.png)

*Figure 1 from [@labanLLMsCorruptYour2026] shows examples of document degradation across different domains*

Surprisingly, the degradation didn't happen gradually over instructions, but models would typically fail catastrophically after a certain number of steps. Better-performing frontier models would fare better only by delaying the step at which the degradation occurs.

They also found that agentic tools don't help prevent degradation; in fact, on average, tool use introduced 6% more degradation and tended to have its own unique failure modes.

## Measuring Document Corruption

To measure document corruption, they introduce a domain-specific document similarity measure that parses documents into components. For a recipe, that means ingredients (name, quantity, unit), steps, and tips; for Python code, it's functions, classes, and imports. That allows them to convert two parsed instances based on the document's true contents and include quantifiable values in the similarity.

![Pipeline diagram showing how a raw recipe text file is parsed into structured ingredients, steps and tips, then scored for semantic equivalence against a reference using a weighted formula: 0.4 times ingredient score plus 0.4 times step score plus 0.2 times tip score.](../../_media/llms-corrupt-figure-5-document-parsing-similarity-score.png)

*Figure 5 from [@labanLLMsCorruptYour2026] - the domain-specific parsing pipeline, with a concrete recipe example showing how ingredients, steps and tips are extracted and compared*

The approach of creating reversible transforms was inspired by [Backtranslation](../../permanent/backtranslation.md), which is an approach of translating and un-translating a document (used in data augmentation and machine translation evaluation), where they would perform some task that transforms a document and then undo it. Imagine splitting a CSV into separate files by expense category, then merging them back together. Or converting all dollar amounts in an accounting ledger to euros, then converting back.

They use a round-trip relay simulation method in which they assume every task is reversible, defined by a forward instruction and its inverse.

![Diagram of the backtranslation round-trip primitive: a seed document s is transformed by a forward edit into document t, then a backward edit reconstructs s-hat, which is compared to s with a similarity function.](../../_media/llms-corrupt-figure-2-backtranslation.png)

*Figure 2 from [@labanLLMsCorruptYour2026] - the backtranslation round-trip primitive: apply a forward edit to get a transformed document, then apply the inverse to reconstruct the original*

They also tested the inclusion of distractor documents in LLM interactions and found that they harm documents more as interaction length increases.

In summary, degradation severity is exacerbated by document size, interaction length, or the presence of distractor files.

### DELEGATE-52

The benchmark consists of seed documents and other content transformed through a sequence of complex editing tasks, designed to resemble the kinds of tasks a worker might delegate to an LLM. The name reflects that tasks are spread across 52 domains.

![Grid of 52 domain icons organised into five colour-coded categories: Code and Configuration (11 domains including Python, Docker, JSON), Science and Engineering (11 domains including Crystal, Molecule, Quantum), Creative and Media (11 domains including Music Sheet, Screenplay, LaTeX), Structured Records (11 domains including Accounting, Genealogy, Spreadsheet), and Everyday (8 domains including Recipe, Chess, Transit).](../../_media/llms-corrupt-figure-3-categories.png)

*Figure 3 from [@labanLLMsCorruptYour2026] - the 52 domains across five categories: Code & Configuration, Science & Engineering, Creative & Media, Structured Records, and Everyday*

In Figure 4, you can see an example of one of the task-sets in the accounting domain.

![Work environment diagram for the accounting domain, showing the Hack Club ledger as the seed document with distractor files including a chart of accounts and expense reimbursement policy. Ten edit tasks branch out, including category split, person split, CSV conversion, euro conversion, and fund accounting, each with a forward and backward instruction.](../../_media/llms-corrupt-figure-4-account-example.png)

*Figure 4 from [@labanLLMsCorruptYour2026] - a work environment from the accounting domain, using a Hack Club ledger as the seed document, with forward/backward edit pairs like splitting by expense category and merging back*

### Results

They tested 19 models across the benchmark. Every single model degraded documents over the course of the simulation. The top performers like Gemini 3.1 Pro, Claude 4.6 Opus, GPT 5.4, still corrupted an average of 25% of document content after 20 interactions. Weaker models averaged 50% degradation.

![Heatmap table of round-trip relay scores for 19 LLMs at workflow lengths 2 through 20. All models show declining scores from left to right, colour-coded from green (high preservation) through yellow to red (severe degradation). Gemini 3.1 Pro scores highest at 80.9 after 20 interactions; GPT 5 Nano scores lowest at 10.0.](../../_media/llms-corrupt-table-1.png)

*Table 1 from [@labanLLMsCorruptYour2026] - round-trip relay results for 19 LLMs across 20 interactions, colour-coded by degradation severity. Every model declines monotonically; frontier models delay but don't avoid the collapse.*

---

One takeaway is that we need to be careful not to extrapolate model capabilities from one area to all domains. Models follow a [Jagged Frontier of LLM Capability](../../permanent/jagged-frontier-of-llm-capability.md) where they can excel in some tasks while making serious errors in others. For example, they perform really well at Python coding and really poorly at 3D object manipulation.

It also raises interesting questions about whether we need to decouple the reasoning engine from the state management system. In a lot of ways, the way LLMs build mini-scripts to perform analysis and manipulate documents is an example of this - LLMs are just not a good solution when you need to retain precise information in memory.
