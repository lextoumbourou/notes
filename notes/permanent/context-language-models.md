---
title: "Context Language Models"
slug: context-language-models
date: 2026-10-15 12:10
modified: 2026-10-18 12:44
category: paper
summary: Language models that manage their own context.
cover: /_media/context-language-models/context-language-models-cover.jpg
hide_cover_in_article: true
bluesky_post: https://bsky.app/profile/notesbylex.com/post/3my4ldahyec2c
mastodon_post: https://fedi.notesbylex.com/@lex/117459571593675764
threads_post: https://www.threads.com/@lexisoninsta/post/Denn-6Mt5Gh
linkedin_post: https://www.linkedin.com/feed/update/urn:li:share:7517373118999474176
x_post: https://x.com/notesbylex_/status/2111648951795605823
tags:
  - ContextManagement
  - LargeLanguageModels
source: https://arxiv.org/abs/2609.37725v1
paper_title: Context Language Models
paper_url: https://arxiv.org/abs/2609.37725v1
paper_authors: Rulin Shao, Shannon Zejiang Shen, Junjie Oscar Yin, Yuetai Li, Minheng Wang, Hamish Ivison, Radha Poovendran, Nathan Lambert, Teng Xiao, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, Pang Wei Koh
paper_year: 2026
code: https://github.com/facebookresearch/context-language-models
---

A paper from the University of Washington and Meta introduces **Context Language Models** (CLMs) [@shaoContextLanguageModels2026].

The idea is to let the model manage its own context, which it receives as a file it can edit across agent turns. They call the technique, aptly, **Context As A File**.

Basically, they mirror the context into storage space and give the model write access. The model can optionally use Bash to edit its context on each turn, and the updated context is provided back to the model on the next turn (keeping some parts intact, like the system prompt).

It's part of a broader movement towards [Thin Harnesses](thin-harnesses.md), handing more and more control over to the model. It seems even the context management policy isn't safe. To quote the authors:

> "Our findings echo [The Bitter Lesson](https://www.incompleteideas.net/IncIdeas/BitterLesson.html) (Sutton, 2019): we should let LMs search for and learn better strategies that go far beyond existing human priors." [@shaoContextLanguageModels2026]

With this newfound freedom, the model came up with some novel context management strategies, like using scoreboards to track its subagents, inventing a "notes" chat role, and writing its own helper for managing compaction. Pretty wild.

Of course, the downside of letting the model edit context is that you may invalidate your KV cache (the model server's saved computation for the context so far) each turn, so the authors also propose a new caching policy to cut that cost, called **[Suffix Cache Reuse](suffix-cache-reuse.md)**.

Out of the box, with no training, it beats existing context-management approaches on deep research and long-running coding tasks and usually uses less compute. They also demonstrate an online RL method for CLMs and show that the model can learn effective context-management strategies and internalise them in its weights.

## How it works

A standard LM appends its output to the context, $c_{t+1} = c_t \oplus f_\theta^{\text{LM}}(c_t)$. A CLM writes the whole next context, $c_{t+1} = f_\theta^{\text{CLM}}(c_t)$. It can keep, cut, rewrite or reorder anything, rather than picking from compaction or offloading actions a human designed.

[![Two formulas on cut-paper strips. A standard language model, on a grey strip, appends its output to the context: c sub t plus 1 equals c sub t concatenated with f LM of c sub t. A context language model, on a pink strip, writes the whole next context: c sub t plus 1 equals f CLM of c sub t. The caption reads: The model decides what its next context is.](../_media/context-language-models/context-language-model-idea.png)](../_media/context-language-models/context-language-model-idea.png)

The [harness](https://github.com/facebookresearch/context-language-models) mirrors the live context into a text file, with a marker like `[[CTX_TURN 4 role=assistant]]` at the start of each turn. The system prompt and task stay pinned outside the file, so the model can't edit away its own instructions. After every command, the harness reads the file back and sends that to the model server, not the accumulated chat history. So deleting lines from the file really does delete them from what the model sees.

[![Two columns show the same context on consecutive turns. On turn 12, the pinned system prompt and task (1.2K tokens) sit above the file LIVE_CTX_MAIN.txt, which holds search results (5.1K), document 58939 (8.0K), a dead-end search (4.9K) and document 21544 (6.3K), 25.5K tokens in total. A pink model block and an arrow labelled "edits the file" run three lines of Python, s = open(ctx).read(), s = compact_turns(s), and open(ctx,"w").write(s). On turn 13, below the same pinned 1.2K, the file holds progress notes (0.6K), three one-line stubs "[Searched: ...]", "[Retrieved doc 58939]" and "[Searched: ...]" (0.1K each), and document 21544 unchanged (6.3K), 8.4K tokens in total.](../_media/context-language-models/context-as-a-file.png)](../_media/context-language-models/context-as-a-file.png)

*An illustrative edit, with made-up token counts. The `compact_turns` helper is modelled on one the model wrote for itself on the [BrowseComp-Plus](browsecomp-plus.md) deep-research benchmark (Figure 3d in [@shaoContextLanguageModels2026]).*

## Multi-agent for free

Since every context is just a file, [Multi-Agent Systems](multi-agent-systems.md) come almost for free. An agent swarm is several context files, each synced to its own model server. Starting a subagent means creating a context file, and killing it means deleting the file. There's no orchestration framework, just more files.

The scoreboard above is one example. The orchestrating agent tracked its subagents with in-place edits to its own context, rather than a tool built for the job.

On Software World, six GPT-5.6-Sol agents with a 272K context limit jointly optimise six interdependent Python repositories for over 24 hours, and are scored on four downstream packages they never see. At the same spend as a summary-based swarm, the CLM swarm gets a 65% greater downstream speedup. Subagents helped less on the single-repository EdgeBench, where CLM with up to five subagents scored 44.2 against 44.6 for CLM on its own.

## ContextBench

They also introduce [ContextBench](contextbench.md), four synthetic tasks that need no reasoning or knowledge, so they test context management alone. CLM stays near 100%. No existing method is perfect, even on these, and even though every method is given a task-specific skill explaining the strategy to use.

[![Four panels illustrate the ContextBench tasks: Needle Retention keeps highlighted lines from several pages, Sudoku Sketchpad edits cells in place on a board, KV Store answers a GET for one key out of many SETs, and Log Triage counts ERROR lines in a log. Below each panel, a line chart plots accuracy against context pressure for Base, RLM, Summary, Context folding, Self-Compact, ACM and CLM. CLM stays at 1.0 across every pressure level on all four tasks, while the other methods fall away as pressure rises.](../_media/context-language-models/contextbench-figure-2.png)](../_media/context-language-models/contextbench-figure-2.png)

*Figure 2 from [@shaoContextLanguageModels2026], CC BY 4.0. All methods use GPT-5.4 with a 32K context limit.*

## Suffix Cache Reuse

Context management usually saves compute anyway, since a shorter context makes every later turn cheaper. They count the re-prefill cost from edits in a metric they call prefix-reuse FLOPs, and CLMs still come out cheaper than the baselines in most experiments.

**Suffix Cache Reuse** (SCR) goes further. If the context `A B C` becomes `A B' C`, standard prefix caching keeps `A` and recomputes `B'` and `C`. SCR also keeps the cached states for `C`, shifts their rotary position encodings (the position information baked into each cached state) to the new positions, and only prefills (computes the cache for) `B'`. It's an approximation, because those states were computed with `B` in front of them. On BrowseComp-Plus, SCR matched the accuracy of standard SGLang (the inference server it's built into) using 65% of its FLOPs.

[![Three rows of blocks show what each cache policy reuses after an edit. Before the edit, blocks A, B and C are all blue, meaning cached. With prefix caching, the context becomes A, a shorter B prime and C; A is blue and reused, while B prime and C are yellow and prefilled again. With Suffix Cache Reuse, A and C are blue and reused, with C labelled "positions shifted", and only B prime is yellow. The caption reads: Prefix caching redoes everything after an edit. SCR keeps the suffix.](../_media/context-language-models/suffix-cache-reuse.png)](../_media/context-language-models/suffix-cache-reuse.png)

## Learning context management

Because context management is now model behaviour, it can be learned in context or in the weights. One sentence in the prompt, like "Back up on disk before you compact.", is enough to change the policy.

For training, they introduce a success-gated efficiency advantage for [Stepwise GRPO](stepwise-grpo.md), which applies the GRPO reinforcement learning method from DeepSeekMath to each step of an agent run. Among the successful runs in a group, cheaper ones get a bonus and more expensive ones a penalty, and failed runs get nothing, so the model can't be rewarded for deleting important information. Qwen3.5-9B goes from 28.8% to 42.5% on BrowseComp-Plus, matching a summary harness trained the same way while using 1.34 PFLOPs per question against 2.19.

## Results

All zero-shot, with no training.

| Task | Setup | Result |
|---|---|---|
| BrowseComp-Plus (deep research) | Qwen3.6-27B, 32K | 59.4%, 11.4% higher (relative) than Codex-style summary, 21.5% fewer FLOPs |
| TerminalBench 2.1 | Qwen3.6-27B, 32K | Matches Codex-style summary with 70% of its FLOPs |
| EdgeBench-10 (12 hours) | Qwen3.6-27B, 32K | 44.6 at 179 PFLOPs against 42.3 at 437 for summary |
| Software World (24 hours, six-agent swarm) | GPT-5.6-Sol, 272K | 65% greater downstream speedup at the same spend |

Codex-style summary, the strongest baseline, summarises the history with Codex's prompts once the context reaches 75% of the budget. EdgeBench reports the best of three seeds, and the CLM gets a reminder 2,048 tokens before it hits its budget.

## Safety

Obviously some safety implications. A context the model can write to is a vector for persistent prompt injections, and for instructions the model writes to itself. The paper points to an OpenAI report where a model inserted unauthorised instructions into its own compaction summary, which then changed how it behaved [@shaoContextLanguageModels2026].

## Further work

The authors suggest distilling harnesses into CLMs. Many harness operations are just transformations of the context, so they could be expressed as CLM actions and eventually trained into the weights.
