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

They introduce a benchmark called [DELEGATE-52](../../permanent/delegate-52.md), which tests whether LLMs can safely carry out long delegated document-editing workflows across 52 domains. Every set of instructions in the benchmark is lossless and reversible, allowing them to measure how much each task degrades the information in the file when the task is reversed.

They found that even the strongest frontier models, including Gemini 3.1 Pro, Claude 4.6 Opus, and GPT 5.4 (the paper was released before their successors), corrupted an average of about 25% of document content after 20 interactions. Across all tested models, average degradation was about 50% [@labanLLMsCorruptYour2026].

Python was the main exception. It was the only domain where most models met the paper’s “delegation-ready” threshold, with 17 of 19 models scoring at least 98% after 20 interactions.

![Three examples of document degradation over 20 interactions: a Linux Kernel Architecture graph diagram losing nodes and edges, a 12-Shaft Twill Diamond textile pattern becoming corrupted, and an ActionBoy Palm Tree 3D object losing geometry. Each shows progressive corruption from interaction 4 to 20.](../../_media/llms-corrupt-figure-1-examples.png)

*Figure 1 from [@labanLLMsCorruptYour2026] shows examples of document degradation across different domains. The benchmark itself is text-only; the visual renderings are illustrative.*

Surprisingly, the degradation didn't happen gradually over instructions, but models would typically fail catastrophically after a certain number of steps. Better-performing frontier models would fare better only by delaying the step at which the degradation occurs.

They also found that tool use did not prevent degradation. The tested models performed worse with tools, averaging an additional 6% of degradation.

## Measuring Document Corruption

To measure document corruption, they introduce a domain-specific [Document Similarity Measure](../../permanent/document-similarity-measure.md) that parses documents into components. For a recipe, that means ingredients (name, quantity, unit), steps, and tips; for Python code, it means functions, classes, and imports. This lets them compare two parsed documents based on their actual content, rather than just raw text. Typical document similarity measures might overlook seemingly small changes, such as `200g` to `800g` of butter, which can be really bad in a recipe, whereas a surface-level rewrite that preserves the underlying structure doesn't need to be heavily penalised.

![Pipeline diagram showing how a raw recipe text file is parsed into structured ingredients, steps and tips, then scored for semantic equivalence against a reference using a weighted formula: 0.4 times ingredient score plus 0.4 times step score plus 0.2 times tip score.](../../_media/llms-corrupt-figure-5-document-parsing-similarity-score.png)

*Figure 5 from [@labanLLMsCorruptYour2026] - the domain-specific parsing pipeline, with a concrete recipe example showing how ingredients, steps and tips are extracted and compared*

The approach of creating reversible transforms was inspired by [Backtranslation](../../permanent/backtranslation.md), a machine translation technique where text is translated into another language and then translated back, allowing the result to be compared with the original. DELEGATE-52 adapts that idea to document editing: apply a forward edit, apply the inverse edit, and compare the reconstructed document to the original. Imagine splitting a CSV into separate files by expense category, then merging them back together. Or converting all dollar amounts in an accounting ledger to euros, then converting them back.

They use a round-trip relay simulation method in which every task is assumed to be reversible, defined by a forward instruction and its inverse.

![Diagram of the backtranslation round-trip primitive: a seed document s is transformed by a forward edit into document t, then a backward edit reconstructs s-hat, which is compared to s with a similarity function.](../../_media/llms-corrupt-figure-2-backtranslation.png)

*Figure 2 from [@labanLLMsCorruptYour2026] - the backtranslation round-trip primitive: apply a forward edit to get a transformed document, then apply the inverse to reconstruct the original*

They also tested the inclusion of distractor documents in LLM interactions and found that they harm documents more as interaction length increases.

Basically, degradation severity is exacerbated by document size, interaction length, and the presence of distractor files.

### DELEGATE-52

The benchmark contains 310 work environments across 52 domains. Each environment includes real seed documents, distractor files, and 5-10 reversible edit tasks that resemble the kinds of tasks a worker might delegate to an LLM.

![Grid of 52 domain icons organised into five colour-coded categories: Code and Configuration (11 domains including Python, Docker, JSON), Science and Engineering (11 domains including Crystal, Molecule, Quantum), Creative and Media (11 domains including Music Sheet, Screenplay, LaTeX), Structured Records (11 domains including Accounting, Genealogy, Spreadsheet), and Everyday (8 domains including Recipe, Chess, Transit).](../../_media/llms-corrupt-figure-3-categories.png)

*Figure 3 from [@labanLLMsCorruptYour2026] - the 52 domains across five categories: Code & Configuration, Science & Engineering, Creative & Media, Structured Records, and Everyday*

Figure 4 shows an example work environment from the accounting domain.

![Work environment diagram for the accounting domain, showing the Hack Club ledger as the seed document with distractor files including a chart of accounts and expense reimbursement policy. Ten edit tasks branch out, including category split, person split, CSV conversion, euro conversion, and fund accounting, each with a forward and backward instruction.](../../_media/llms-corrupt-figure-4-account-example.png)

*Figure 4 from [@labanLLMsCorruptYour2026] - a work environment from the accounting domain, using a Hack Club ledger as the seed document, with forward/backward edit pairs like splitting by expense category and merging back*

### Results

They tested 19 models across the benchmark. All 19 models degraded documents over the course of the simulation. The top performers, such as Gemini 3.1 Pro, Claude 4.6 Opus, and GPT 5.4, still corrupted an average of about 25% of the document content after 20 interactions. Across all tested models, average degradation was about 50%, with weaker models failing more severely.

![Heatmap table of round-trip relay scores for 19 LLMs at workflow lengths 2 through 20. All models show declining scores from left to right, colour-coded from green (high preservation) through yellow to red (severe degradation). Gemini 3.1 Pro scores highest at 80.9 after 20 interactions; GPT 5 Nano scores lowest at 10.0.](../../_media/llms-corrupt-table-1.png)

*Table 1 from [@labanLLMsCorruptYour2026] - round-trip relay results for 19 LLMs across 20 interactions, colour-coded by degradation severity. Every model declines over time; frontier models delay but do not avoid degradation.*

Short-term performance did not reliably predict long-horizon performance. Some models that looked similar after two interactions diverged sharply after twenty, while others that started behind later caught up. This is one of the reasons the paper argues for long-horizon evaluation rather than only testing one-shot or short workflows.

The kind of degradation also changes with model strength. Weaker models tend to lose content through deletion, while frontier models are more likely to preserve content but corrupt it.

The paper also finds that some document types are much harder than others. LLMs degrade less on repetitive, numerical, and structurally dense formats, and more on natural-language-heavy or lexically diverse documents. Tasks involving global restructuring, such as splitting, merging, and classification, are harder than more local operations, such as string manipulation or referencing.

---

One takeaway is that we need to be careful not to extrapolate model capabilities from one area to all domains. Models follow a [Jagged Frontier of LLM Capability](../../permanent/jagged-frontier-of-llm-capability.md), where they can excel in some tasks while making serious errors in others. For example, they perform well on Python and poorly on some structured-but-unfamiliar document formats, such as textual 3D object files.

It also raises interesting questions about whether we need to decouple the reasoning engine from the state management system. LLMs may be useful as the reasoning layer, but long-running document workflows probably need external state, parsers, validators, diffs, tests, and reversible operations to prevent silent corruption.