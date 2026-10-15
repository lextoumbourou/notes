---
title: Context Language Models
date: 2026-10-15 12:10
modified: 2026-10-16 08:06
status: draft
category: paper
summary: Context Language Models let the model learn its own context management policy, by treating its context as a file it can edit.
cover: /_media/context-language-models/archimedes-palimpsest-cover.jpg
hide_cover_in_article: true
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

A 2 week old paper from the University of Washington and Meta, that introduces **Context Language Models** (CLMs) [@shaoContextLanguageModels2026].

The idea is to let the model learn its own Context Management Policy. They do it with a Context As A File mechanism, where they mirror the context into storage space with LM write access. The LM can append newly generated tokens or use Bash to edit the context, and each edit is synchronised to the live context for the next turn.

Out of the box, with no training, it beats existing context management approaches on deep research and long-running coding tasks, and usually uses less compute. On BrowseComp-Plus, Qwen3.6-27B with a 32K context limit scores 59.4%, which is 11.4% higher than Codex-style summarisation, with 21.5% fewer FLOPs.

The obvious catch is the KV cache. Editing the middle of your context means recomputing everything after it, so they also introduce a new caching policy to cut that cost.

## How it works

Most harnesses manage context with a fixed policy. Codex and Cursor compact when the context reaches a set length, and the model writes a summary that replaces the history. Newer methods give the model some control through tools. Self-Compact and AutoCompact let the model decide when to compact, and ACM lets it offload context to storage and fetch it back. But the model can still only pick from actions a human designed.

CLMs drop the action list. A standard LM appends its output to the context, $c_{t+1} = c_t \oplus f_\theta(c_t)$. A CLM writes the whole next context, $c_{t+1} = f_\theta(c_t)$. The next context is whatever the model decides. It can keep, cut, rewrite or reorder anything.

The harness mirrors the live context into a text file (`/tmp/.live_ctx/LIVE_CTX_MAIN.txt`) and puts the path in the system prompt. The system prompt and the task stay pinned outside the file, so the model can't edit away its own instructions. Everything after them goes in the file, and each turn starts with a marker like `[[CTX_TURN 4 role=assistant]]`. The model edits it with Bash or Python, like any other file, and every change is synced back into what the model sees on the next turn. If it doesn't touch the file, the new tokens are appended as normal.

So when the paper says "zero-shot" or "out of the box", it means an existing model with unchanged weights, a Bash tool and a system prompt telling it about the file. The bit that isn't just a Bash tool is the sync. With a normal Bash tool, the model can write notes to a file, but that only adds to its context. Nothing it does can take an old search result back out of the conversation. Here the file is the prompt. After every command the harness reads the file back and sends that to the model server, not the accumulated chat history, so deleting lines from the file really does delete them from what the model sees.

[![Two columns show the same context on consecutive turns. On turn 12, the pinned system prompt and task (1.2K tokens) sit above the file LIVE_CTX_MAIN.txt, which holds search results (5.1K), document 58939 (8.0K), a dead-end search (4.9K) and document 21544 (6.3K), 25.5K tokens in total. A pink model block and an arrow labelled "edits the file" run three lines of Python, s = open(ctx).read(), s = compact_turns(s), and open(ctx,"w").write(s). On turn 13, below the same pinned 1.2K, the file holds progress notes (0.6K), three one-line stubs "[Searched: ...]", "[Retrieved doc 58939]" and "[Searched: ...]" (0.1K each), and document 21544 unchanged (6.3K), 8.4K tokens in total.](../_media/context-language-models/context-as-a-file.png)](../_media/context-language-models/context-as-a-file.png)

*An illustrative edit, with made-up token counts. The `compact_turns` helper is modelled on one the model wrote for itself on BrowseComp-Plus (Figure 3d in [@shaoContextLanguageModels2026]).*

Since every context is just a file, multi-agent comes for free. An agent swarm is several context files, each synced to its own model server. Starting a subagent means creating a context file, and killing it means deleting the file.

Given that freedom, the model came up with some strategies of its own (Figure 3):

- On a maths problem with subagents, it kept a scoreboard of their status and updated it with 163 in-place edits, while keeping its own context at 6 to 8K tokens.
- It invented a new chat role, `notes`, for its own internal notes. In the released harness, any role it doesn't recognise is sent to the model as a user message, so this is really a user turn with a new label.
- It wrote for loops that replace old search results with one-line stubs like `[Searched: query]`.
- It defined a `compact_turns` helper function and reused it 37 times.
- It also rediscovered the usual summary trick, squeezing 21K tokens into a summary of only the facts relevant to the answer.

## In code

The authors released their harness on [GitHub](https://github.com/facebookresearch/context-language-models). The core of it is small enough to sketch in about 40 lines. This is my simplified version, modelled on their [context_env/env.py](https://github.com/facebookresearch/context-language-models/blob/main/clm/clm_harness/context_env/env.py) and [context_utils/context_string.py](https://github.com/facebookresearch/context-language-models/blob/main/clm/clm_harness/context_utils/context_string.py).

```python
import os, re, subprocess

CTX = "/tmp/.live_ctx/LIVE_CTX_MAIN.txt"
HEADER = re.compile(r"^\[\[CTX_TURN \d+ role=(\w+)\]\]$", re.M)


def render(messages):
    # Everything after the system prompt and task, one block per turn.
    return "\n\n".join(
        f"[[CTX_TURN {i} role={m['role']}]]\n{m['content']}"
        for i, m in enumerate(messages[2:], start=1)
    )


def parse_back(text, pinned):
    # Whatever headers survive the edit become the new turns.
    parts = HEADER.split(text)[1:]  # [role, body, role, body, ...]
    turns = [
        {"role": "assistant" if role == "assistant" else "user", "content": body.strip()}
        for role, body in zip(parts[::2], parts[1::2])
        if body.strip()
    ]
    return pinned + turns


def run(messages, llm):
    os.makedirs(os.path.dirname(CTX), exist_ok=True)
    while True:
        command = llm(messages)  # the model replies with one bash command
        before = render(messages)
        with open(CTX, "w") as f:  # 1. mirror the context into the file
            f.write(before)
        out = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
        with open(CTX) as f:  # 2. read it back after the command
            after = f.read()
        if after != before:  # 3. the edited file becomes the context
            messages = parse_back(after, messages[:2])
        if "SUBMIT" in command:
            return messages
        messages += [
            {"role": "assistant", "content": command},
            {"role": "tool", "content": out},
        ]
```

Step 3 is the whole idea. If the command changed the file, `messages` is replaced by whatever is in the file now, and that's what the model gets on the next call.

On the model's side, an edit is just a Bash command. This is the example their system prompt gives the model, which collapses turn 7 into a one-line note without retyping it:

```bash
python3 - <<'PY'
import re; p="/tmp/.live_ctx/LIVE_CTX_MAIN.txt"; s=open(p).read()
# collapse turn 7 (its header -> the next header); its body is never retyped:
s=re.sub(r"(\[\[CTX_TURN 7 [^\]]*\]\]).*?(?=\n\[\[CTX_TURN|\Z)",
         r"\1\n[grep done: parser.py:142 drops quoted commas; fix=csv.reader]", s, flags=re.S)
open(p,"w").write(s)
PY
```

The real harness does more than my sketch. It shows a token count after every command, nudges the model at 25%, 50% and 75% of its budget, rolls back turns that would overflow, and rejects edits that would push the context over the limit. An edit that changes the file and prints nothing doesn't count as a turn.

The system prompt also does a lot of the work. It tells the model to replace "big outputs, dead ends, superseded notes" with a short summary, and it explains the cache cost, saying "an edit forces everything *after* it to be re-read, so cost grows with how much text FOLLOWS the edit". So the zero-shot model isn't working out its context management alone. It's been told how the file works and what an edit costs.

## ContextBench

They conduct a pilot study with 4 synthetic tasks, which they call ContextBench. The tasks need no reasoning or knowledge, so a model that could keep everything in context would answer every query. That isolates context management from everything else. The context limit is fixed at 32K and they push up to 24 times that much input through it.

- **Needle Retention** asks the model to keep specific lines word for word while deleting filler.
- **Sudoku Sketchpad** asks it to keep a 16×16 Sudoku board up to date as moves stream in, one per turn.
- **KV Store** sends batches of 100 key-value pairs, then asks for exact values later.
- **Log Triage** streams service logs, then asks lookup and count questions about them.

Answers have to be in the context. A value sitting in a file on disk doesn't count.

With GPT-5.4, none of the existing methods were perfect, even on these simple tasks. Summaries lose or hallucinate needles and Sudoku cells. Methods without in-place editing have to regenerate the whole board for every move. Coding tools can offload values to disk but can't evict them from the live context. CLM stays near 100% at every level of pressure (Figure 2).

One thing worth knowing is that every method gets a task-specific skill explaining how to use its own tools on each task. The CLM's KV Store skill tells it to move each batch out of its context into a file and grep for it later. So ContextBench tests whether a method can carry out a strategy, not whether the model can come up with one.

## The KV cache problem

One issue is that if you're editing your context, you're invalidating the previous cache, right?

Serving engines like SGLang reuse the cached keys and values for the longest prefix that matches the previous request. Appending keeps the whole prefix. An edit in the middle forces everything after the edit to be prefilled again. In their worked example for Qwen3.6-27B (Appendix C), a 20K-token prompt with a 500-token response costs 1.41×10¹⁴ FLOPs when the turn only appends, 5.74×10¹⁴ after an edit in the middle, and 7.7 times the append-only cost after an edit at the start.

Well kinda. But context management can be made compute efficient generally, since a shorter context makes every later turn cheaper. They count the re-prefill cost in a metric they call prefix-reuse FLOPs, and CLMs still come out cheaper than the baselines in most experiments.

They also introduce a new caching policy, **Suffix Cache Reuse** (SCR). If the context `A B C` becomes `A B' C`, standard prefix caching keeps `A` and recomputes `B'` and `C`. SCR also keeps the cached states for `C`, shifts their rotary position encodings to the new positions, and only prefills `B'`. It's an approximation, because those cached states were computed with `B` in front of them, not `B'`. They cap it at six relocated spans per edit, so the approximations can't pile up.

On BrowseComp-Plus, SCR matched standard SGLang's accuracy (60.2%) using 65% of its FLOPs. A nice bonus is that it helps normal chat serving too. Reasoning models' chat templates strip the thinking from earlier turns, which is also a mid-context edit. Most of SCR's extra cache reuse (5.3 of 7.8 percentage points of prompt tokens) came from that, not from the model's own edits.

## Learning context management

Because context management is now model behaviour rather than harness code, it can be learned like any other skill, either in context or in the weights.

The in-context version is simple. You can just tell it. One sentence added to the prompt, like "Before every edit to your context, you must back up the current context file", changed when it compacted, where it compacted and whether it backed up first. They also evolve a skill document with a prompt-evolution loop, where a proposer model reads rollouts and rewrites the skill. That improved held-out accuracy on ContextBench by up to 35.9 points while reducing compute.

For training, they introduce a success-gated efficiency advantage for Stepwise GRPO.

[Group Relative Policy Optimisation](group-relative-policy-optimisation.md) samples a group of complete trajectories for the same prompt and scores each one against the group average. A CLM trajectory isn't one long sequence, since edits change the input from turn to turn. Stepwise GRPO keeps each model call with the context it actually saw, and gives every call in a trajectory that trajectory's advantage.

The outcome reward alone is a weak signal for editing. Successful runs can contain wasteful edits and failed runs can contain useful ones. Rewarding edits directly, or rewarding tokens removed, invites reward hacking, since the model could delete important information or edit for the sake of it. So the efficiency bonus only applies among the successful trajectories in a group. Successful runs that used fewer prefix-reuse FLOPs than the successful average get a bonus, more expensive ones get a penalty, and failed runs get nothing either way. If fewer than two runs in the group succeed, there's no efficiency signal at all.

They post-train Qwen3.5-9B on OpenResearcher and test on BrowseComp-Plus. Before training, the small model is a weaker CLM than the summary harness (28.8% against 34.7%). After RL with the same recipe, CLM reaches 42.5% and the summary harness 42.1%, but CLM uses 1.34 PFLOPs per question against 2.19.

## Results

All of these are zero-shot, with no training.

| Task | Setup | Result |
|---|---|---|
| BrowseComp-Plus (deep research) | Qwen3.6-27B, 32K | 59.4%, 11.4% higher than Codex-style summary, 21.5% fewer FLOPs |
| TerminalBench 2.1 | Qwen3.6-27B, 32K | Matches Codex-style summary with 70% of its FLOPs |
| TBLite | Qwen3.6-27B, 32K | 73.7% against 67.0% for Codex-style summary |
| Maths optimisation (4 problems) | Claude 4.6 Sonnet | Beats the OpenEvolve evolutionary workflow on all four |
| EdgeBench-10 (12 hours) | Qwen3.6-27B, 32K | 44.6 at 179 PFLOPs against 42.3 at 437 for summary |
| EdgeBench-10 (12 hours) | Claude 4.6 Sonnet, 32K | 51.0 against 42.3 for summary |
| Software World (24 hours, six-agent swarm) | GPT-5.6-Sol, 272K | 65% greater downstream speedup at the same spend |

Subagents added little on single-repository EdgeBench.

A few caveats. The maths problems are one run per method, and EdgeBench reports the best of three seeds. The CLM gets a reminder 2,048 tokens before it hits its budget, because models are bad at knowing how full their own context is (Appendix G). On BrowseComp-Plus, CLM's editing turns don't count towards the 100-turn limit.

## Safety

Obviously some safety implications. A context the model can write to is a vector for persistent prompt injections, and for instructions the model writes to itself. The paper points to an OpenAI report where a model inserted unauthorised instructions into its own compaction summary, which then changed how it behaved.

## Further work

- Scale RL training so CLMs can explore and learn context management strategies.
- Develop a harness-to-CLM pipeline that distils harnesses into CLMs.

The second one is the interesting framing. Many harness operations are just transformations of the context, so they could be expressed as CLM actions and eventually trained into the weights. The authors describe harnesses as a kind of procedural memory that a CLM could absorb.
