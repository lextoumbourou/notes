---
title: "LLMs Corrupt Your Documents When You Delegate"
date: 2026-05-09 00:00
modified: 2026-05-09 00:00
category: reference/papers
cover: /_media/llms-corrupt-cover.png
hide_cover_in_article: true
summary: "A large-scale study on long-horizon document tasks."
tags:
- AgenticReasoning
- LimitationsofLLMs

---

*My notes on [LLMs Corrupt Your Documents When You Delegate](https://arxiv.org/pdf/2604.15597) by Philippe Laban, Tobias Schnabel and Jennifer Neville from Microsoft Research.*

An interesting paper from 3 researchers at Microsoft.

They found that when delegating document manipulation tasks to an LLM (things like splitting a CSV into separate files based on categories, transposing the key of a music score) in almost all cases, the information in the document would degrade over time. Even with the latest frontier models, users would see an average of 25% document corruption by the end of a long workflow, with all tested models averaging 50% corruption [@labanLLMsCorruptYour2026].

Surprisingly, this isn't something that gradually decays over time - models would typically fail catastrophically after a certain number of tasks, with frontier models just delaying the step at which the degradation occurs.

Interestingly, they also found that agentic tools don't help prevent degradation and tend to have their own unique failure modes.

![Three examples of document degradation over 20 interactions: a Linux Kernel Architecture graph diagram losing nodes and edges, a 12-Shaft Twill Diamond textile pattern becoming corrupted, and an ActionBoy Palm Tree 3D object losing geometry. Each shows progressive corruption from interaction 4 to 20.](../../_media/llms-corrupt-figure-1-examples.png)

*Figure 1 from [@labanLLMsCorruptYour2026] shows examples of document degradation across different domains*

## Measuring Document Corruption

How did they measure document corruption?

Firstly, they introduce a domain-specific document similarity measure that parses documents into components. For a recipe, that means ingredients (name, quantity, unit), steps, and tips; for Python code, it's functions, classes, and imports.

![Pipeline diagram showing how a raw recipe text file is parsed into structured ingredients, steps and tips, then scored for semantic equivalence against a reference using a weighted formula: 0.4 times ingredient score plus 0.4 times step score plus 0.2 times tip score.](../../_media/llms-corrupt-figure-5-document-parsing-similarity-score.png)

*Figure 5 from [@labanLLMsCorruptYour2026] - the domain-specific parsing pipeline, with a concrete recipe example showing how ingredients, steps and tips are extracted and compared*

Then they used a [Backtranslation](../../permanent/backtranslation.md)-inspired approach - which is an approach of translating and un-translating a document (used in data augmentation and machine translation evaluation), where they would perform some task that transforms a document and then undo it. Imagine splitting a CSV into separate files by expense category, then merging them back together. Or converting all dollar amounts in an accounting ledger to euros, then converting back.

They use a round-trip relay simulation method in which they assume every task is reversible, defined by a forward instruction and its inverse.

![Diagram of the backtranslation round-trip primitive: a seed document s is transformed by a forward edit into document t, then a backward edit reconstructs s-hat, which is compared to s with a similarity function.](../../_media/llms-corrupt-figure-2-backtranslation.png)

*Figure 2 from [@labanLLMsCorruptYour2026] - the backtranslation round-trip primitive: apply a forward edit to get a transformed document, then apply the inverse to reconstruct the original*

They also tested distractor documents and found that they harm documents more as the interaction length increases.

Degradation severity is exacerbated by document size, interaction length, or the presence of distractor files.

### DELEGATE-52

The paper also contributes a benchmark called [DELEGATE-52](../../permanent/delegate-52.md) which demonstrated [Long-Horizon Workflow](../../permanent/long-horizon-workflow.md) tasks requiring in-depth document editing across 52 professional domains (see Figure 3).


The benchmark consists of seed documents and other content transformed through a sequence of complex editing tasks, designed to resemble the kinds of tasks a worker might delegate to an LLM.

![Grid of 52 domain icons organised into five colour-coded categories: Code and Configuration (11 domains including Python, Docker, JSON), Science and Engineering (11 domains including Crystal, Molecule, Quantum), Creative and Media (11 domains including Music Sheet, Screenplay, LaTeX), Structured Records (11 domains including Accounting, Genealogy, Spreadsheet), and Everyday (8 domains including Recipe, Chess, Transit).](../../_media/llms-corrupt-figure-3-categories.png)

*Figure 3 from [@labanLLMsCorruptYour2026] - the 52 domains across five categories: Code & Configuration, Science & Engineering, Creative & Media, Structured Records, and Everyday*

Example of one of the documents, with its tasks.

![Work environment diagram for the accounting domain, showing the Hack Club ledger as the seed document with distractor files including a chart of accounts and expense reimbursement policy. Ten edit tasks branch out, including category split, person split, CSV conversion, euro conversion, and fund accounting, each with a forward and backward instruction.](../../_media/llms-corrupt-figure-4-account-example.png)

*Figure 4 from [@labanLLMsCorruptYour2026] - a work environment from the accounting domain, using a Hack Club ledger as the seed document, with forward/backward edit pairs like splitting by expense category and merging back*

### Results

They tested 19 models across the benchmark. Every single model degraded documents over the course of the simulation. The top performers like Gemini 3.1 Pro, Claude 4.6 Opus, GPT 5.4, still corrupted an average of 25% of document content after 20 interactions. Weaker models averaged 50% degradation.

![Heatmap table of round-trip relay scores for 19 LLMs at workflow lengths 2 through 20. All models show declining scores from left to right, colour-coded from green (high preservation) through yellow to red (severe degradation). Gemini 3.1 Pro scores highest at 80.9 after 20 interactions; GPT 5 Nano scores lowest at 10.0.](../../_media/llms-corrupt-table-1.png)

*Table 1 from [@labanLLMsCorruptYour2026] - round-trip relay results for 19 LLMs across 20 interactions, colour-coded by degradation severity. Every model declines monotonically; frontier models delay but don't avoid the collapse.*

Python was the only domain where models were genuinely ready for delegation, with 17 of 19 models achieving near-lossless manipulation. Outside of that, models were considered "ready" (>=98% reconstruction score) in fewer than 11 of 52 domains, even for the best model.

The failure mode is also worth noting: it's not a slow bleed. Models maintain near-perfect reconstruction for several rounds, then experience a sudden catastrophic failure, typically losing 10–30 points in a single round-trip. These sparse critical failures account for about 80% of the total observed degradation.

---

One takeaway is that we need to be careful not to extrapolate model capabilities from one area to all domains. Models follow a [Jagged Frontier of LLM Capability](../../permanent/jagged-frontier-of-llm-capability.md) where they can excel in some tasks while making serious errors in others. In particular, they perform really well at Python coding and really poorly at 3D object manipulation.

It also raises interesting questions about whether we need to decouple the reasoning engine from the state management system. In a lot of ways, the way LLMs build mini-scripts to perform analysis and manipulate documents is an example of this - LLMs are just not a good solution when you need to retain precise information in memory.
