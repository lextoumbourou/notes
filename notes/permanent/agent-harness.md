---
title: Agent Harness
date: 2026-10-08 13:50
modified: 2026-10-08 16:46
status: draft
summary: Basically just while loops.
tags:
- HarnessDesign
- AgenticReasoning
---

The **agent harness** is everything in [AI Agents](ai-agents.md) that isn't the model.

It typically operates as a loop where:

- The harness prepares the model context.
- The harness calls the LLM with the context.
- The LLM decides on whether to call a tool or end.
- Some tool is called, and the results are fed into context.

The core it is something like these few lines of Python: 

```python
while True:
    reply = model(context)
    if not reply.tool_call:
        break
        
    result = run_tool(reply.tool_call)
    context.append(result)
```

Of course, real harnesses turn out to be a lot bigger than that, because have to do a bunch of extra stuff, like safety checks, support extensions, orchestiation.

A study of eleven production coding harnesses, including Claude Code, Codex CLI, Gemini CLI and Pi, lists about 7 parts to the typically harness project [@barbasteHarnessEngineeringAnatomy2026]:

1. the loop
2. the LLM integration
3. tools
4. context management
5. safety controls
6. orchestration (running several agents or tasks together)
7. extension surfaces (places users can plug in their own tools and skills)

The paper closes with a 90-line minimum viable harness. 90 lines isn't much, so I'm literally going to test that in this blog post, with a real-call to OpenAI.

For coding agents specifically, the harness is called a [Coding Harness](coding-harness.md) - although most agentic harness seem to be coding harnesses these days.

The same eleven-harness study found that none of them imports a general-purpose agent framework: they all run hand-written loops [@barbasteHarnessEngineeringAnatomy2026]. Studying the same harnesses again a quarter later, the authors saw them converging, to the point of copying each other.

At the small end is Pi, Mario Zechner's coding agent. It gives the model four tools (read, write, edit and bash), and its system prompt and tool definitions together come to under 1,000 tokens [@zechnerWhatLearnedBuilding2025]. Pi reached 1.0 on 1 October 2026 and its makers still describe it as minimal [@earendilPi102026].

Docker Agent goes further and makes the agent a configuration file [@dockerDockerAgent]. An agent is a model, a description, an instruction and a list of toolsets, written in YAML.

## "Thin" Harnesses

Best practices for harness engineering just 6 months ago 

Most of the advice from people who build harnesses points the same way: keep it small.

Garry Tan's version is "thin harness, fat skills" [@tanThinHarnessFat2026]. He limits the harness to four jobs: run the model in a loop, read and write files, manage context and enforce safety. Everything else goes either up, into Markdown skill files that describe a process, or down, into fast deterministic tools. His anti-pattern is the fat harness, with dozens of tool definitions eating the context window.

Philipp Schmid's talk builds the same GitHub pull-request reviewer three times [@schmidAgentsCodeSkills]. The first version is a hand-written Python loop with a JSON schema for every tool. The second uses an agent framework that hides the loop. The third has no source directory at all: an `AGENTS.md` file, a script that installs the GitHub CLI, and a sandbox where the model can run bash. Each version deletes code from the one before, and the last one can also answer questions the first could not, because it is not limited to the tools someone wrote for it. His rule of thumb: "If your harness is getting more complex as the model improves, you are most likely overengineering your harness."

Behind both is Rich Sutton's essay "The Bitter Lesson": "general methods that leverage computation are ultimately the most effective, and by a large margin" [@suttonBitterLesson2019]. Building knowledge into a system helps in the short term, then plateaus. A lot of what goes into a harness is that kind of knowledge: plans, checklists and special-purpose tools that make up for what the model can't yet do.

## The harness still matters

If harnesses are converging and thin is best, you might expect the harness to make little difference. Two recent papers found that it still does.

*Don't Blame the Large Language Model* held the model fixed and ran 35 consecutive releases of the Qwen Code CLI against the same 50 SWE-bench Verified tasks, which are real GitHub issues the agent has to fix [@sghaierDontBlameLarge2026]. The resolve rate moved between 23% and 39% around a mean of 30.5%, with no statistically significant improvement, while tokens per task rose by more than 70%. The authors traced individual shifts to specific pull requests, and every regression they documented had passed the project's automated checks. Practitioners, they note, tend to blame the model for regressions the harness caused.

[An Empirical Study of Harness Design for Coding Agents](an-empirical-study-of-harness-design-for-coding-agents.md) kept the loop fixed and changed one component at a time [@fanEmpiricalStudyHarness2026]:

- Planning improved accuracy for weaker models, but mostly saved cost for stronger ones.
- Predefined file tools helped models that were weaker at bash. Models that were good at bash did well with bash alone, at much lower cost.
- [Context Management](context-management.md) mattered more as the context window shrank, mostly by preventing overflow failures.

So the builders and the researchers aren't really in conflict. The best harness depends on the model, so pieces that helped last year's model can be dead weight for this year's. That is why "build to delete" works.

## Where the scaffolding goes

When a piece comes out of the loop, it usually ends up somewhere else.

**Into skills.** An [Agent Skill](agent-skill.md) is a folder with a `SKILL.md` file of instructions and any scripts it needs, loaded only when the task calls for it. In the eleven-harness study, skills were more widely adopted than [MCP](mcp.md), the Model Context Protocol for connecting agents to external tool servers: nine of the eleven harnesses support them, against eight for MCP [@barbasteHarnessEngineeringAnatomy2026]. Project instructions move into [Agent Context Files](agent-context-files.md) such as `AGENTS.md`.

**Into configuration.** The same study found behavioural rules moving out of system-prompt prose and into configuration [@barbasteHarnessEngineeringAnatomy2026].

**Into the model.** Harness-Zero uses a specialised harness only during training, then fine-tunes the model (trains it further) on the corrected runs so the harness can be removed [@yeHarnessZeroHarnessDistillation2026]. On their benchmarks, the distilled model's average task success went from 23.3% to 44.3%, beating the 41.7% the base model reached with the harness still attached.

**Into other harnesses.** Harnesses are starting to host each other. Six of the eleven harnesses ship the Agent Client Protocol (ACP), an editor protocol, and OpenHands now uses it to run rival harnesses as "interchangeable brains" inside its own conversations [@barbasteHarnessEngineeringAnatomy2026]. Docker Agent can hand its loop to Claude Code, Codex, OpenCode or Pi, while keeping its own orchestration, hooks and permissions [@dockerDockerAgent]. Raven, which calls itself "the harness of harnesses", builds and evolves a separate harness for each model and domain, then has a host agent split a goal between them [@evermindRavenHarnessHarnesses2026].

Some researchers go a step further and let an agent edit its own harness. RRSI found that doing this freely on a fixed set of tasks tends to overfit: gains on the tuning tasks don't carry over to new ones, so they constrain which edits get proposed and kept [@xiaRRSIRegularizedRecursive2026].

## What stays

Some parts of a harness are not compensating for a weak model, so a better model won't remove them:

- **Sandboxing and permissions.** Docker Agent's sandbox mode runs the agent in a virtual machine that can only see the working directory, with outbound network access blocked except for an allowlist [@dockerDockerAgent]. In Schmid's talk, the sandbox's network proxy adds credentials to outgoing requests, so the agent can call GitHub without ever seeing the token [@schmidAgentsCodeSkills].
- **Context management.** Context windows are finite, and long tasks still overflow them [@fanEmpiricalStudyHarness2026].
- **Evals.** Evals are the automated tests that check whether an agent still does the job after a change. Schmid's advice is to own your instructions, your workflows and especially your evals [@schmidAgentsCodeSkills]. He also argues that the runs a harness records become valuable training and evaluation data: "the Harness is the Dataset" [@schmidImportanceAgentHarness2026].

## Testing the 90 lines

<!-- Drafted by Claude Code on 8 October 2026 for Lex to rewrite in his own words. The code and the run's output are verbatim. -->

Listing 3 of the eleven-harness paper is a "minimum viable harness" in about 90 lines of Python [@barbasteHarnessEngineeringAnatomy2026]. The authors call it an illustrative scaffold, not production code, and as printed it doesn't run:

- `Model` is an interface with no implementation, so there's nothing to call.
- Each tool schema has a name and a description but no parameters, so the model can't know what arguments to send.
- The turn and cost limits raise `StopIteration`. Inside an `async` function, Python turns that into a `RuntimeError` ([PEP 479](https://peps.python.org/pep-0479/)), so hitting a limit crashes the agent instead of stopping it.

To run it against [GPT-6 Luna](gpt-6-luna.md), I added an OpenAI client, generated parameter schemas from the tool functions, swapped `StopIteration` for an exception of its own, read tool calls in OpenAI's shape, stopped compaction from leaving a tool result without the call that asked for it, and fixed the line numbers `read_file` reports. Function calling through Chat Completions on GPT-6 Luna needs `reasoning_effort="none"`. Every changed line is marked `# +`. It came to 129 lines, 106 of them code.

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0"]
# ///
"""Listing 3 of Barbaste et al. (2026), made runnable against OpenAI.
CC BY 4.0. Lines marked "# +" were added or changed to make it run."""
from __future__ import annotations
import asyncio, json, pathlib, subprocess, sys
from dataclasses import dataclass, field
from typing import Protocol
from openai import AsyncOpenAI  # +

class Model(Protocol):
    async def complete(self, messages: list[dict],
                       tools: list[dict]) -> dict: ...

class OpenAIModel:  # + the paper leaves the provider to you
    def __init__(self, name="gpt-6-luna"):
        self.client, self.name = AsyncOpenAI(), name
    async def complete(self, messages, tools):
        r = await self.client.chat.completions.create(
            model=self.name, messages=messages, tools=tools or None,
            reasoning_effort="none")
        m, u = r.choices[0].message, r.usage
        out = {"role": "assistant", "content": m.content or "",
               "cost": u.prompt_tokens * 0.10e-6
                       + u.completion_tokens * 0.50e-6}
        if m.tool_calls:
            out["tool_calls"] = [c.model_dump() for c in m.tool_calls]
        return out

class LimitReached(Exception): ...  # + StopIteration breaks in async

def _truncate(s: str, n: int = 25_000) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncated]"

def tool_bash(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=120)
    return _truncate(f"exit={r.returncode}\nstdout:\n{r.stdout}"
                     f"\nstderr:\n{r.stderr}")

def tool_read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    lines = pathlib.Path(path).read_text().splitlines()
    return _truncate("\n".join(f"{i + 1:4}: {ln}" for i, ln in
                     enumerate(lines[offset:offset + limit], offset)))  # +

def tool_write_file(path: str, content: str) -> str:
    pathlib.Path(path).write_text(content)
    return f"wrote {len(content)} bytes"

def tool_search_replace(path: str, search: str, replace: str) -> str:
    p = pathlib.Path(path); text = p.read_text()
    if text.count(search) != 1:
        return f"ERROR: search string occurs {text.count(search)}x"
    p.write_text(text.replace(search, replace, 1)); return "OK"

TOOLS = {"bash": tool_bash, "read_file": tool_read_file,
         "write_file": tool_write_file,
         "search_replace": tool_search_replace}

def schema(name, f):  # + the model needs each tool's parameters
    args = f.__code__.co_varnames[:f.__code__.co_argcount]
    props = {a: {"type": "integer" if a in ("offset", "limit")
                 else "string"} for a in args}
    required = [a for a in args if a not in ("offset", "limit")]
    return {"type": "function", "function": {"name": name,
            "parameters": {"type": "object", "properties": props,
                           "required": required}}}

def discover_context(cwd: pathlib.Path = pathlib.Path.cwd()) -> str:
    parts = []
    for p in reversed([cwd, *cwd.parents]):  # root to leaf
        md = p / "AGENTS.md"
        if md.exists():
            parts.append(f"<ctx path='{md}'>\n{md.read_text()}\n</ctx>")
    return "\n".join(parts)

@dataclass
class Agent:
    model: Model
    messages: list = field(default_factory=list)
    n_turns: int = 0
    cost: float = 0.0
    max_turns: int = 50
    max_cost: float = 5.00
    compact_at_tokens: int = 120_000

    def _token_estimate(self) -> int:
        return sum(len(json.dumps(m)) for m in self.messages) // 4

    async def _check_limits(self):
        if self.n_turns >= self.max_turns: raise LimitReached("turns")
        if self.cost >= self.max_cost: raise LimitReached("cost")

    async def _maybe_compact(self):
        if self._token_estimate() < self.compact_at_tokens: return
        summary = await self.model.complete(self.messages + [
            {"role": "user", "content": "Summarize the conversation. "
             "Preserve decisions and unresolved issues."}], tools=[])
        keep = max(4, int(len(self.messages) * 0.30))
        tail = self.messages[-keep:]
        while tail and tail[0]["role"] == "tool": tail = tail[1:]  # +
        self.messages = [self.messages[0], {"role": "assistant",
                         "content": summary["content"]}] + tail

    async def run(self, task: str) -> str:
        self.messages = [
            {"role": "system", "content":
             f"You are a SWE agent.\n\n{discover_context()}"},
            {"role": "user", "content": task}]
        tool_schemas = [schema(n, f) for n, f in TOOLS.items()]  # +
        while True:
            await self._check_limits(); await self._maybe_compact()
            resp = await self.model.complete(self.messages, tool_schemas)
            self.n_turns += 1; self.cost += resp.pop("cost", 0.0)  # +
            self.messages.append(resp)
            if not resp.get("tool_calls"): return resp["content"]
            for call in resp["tool_calls"]:  # serial execution
                fn = call["function"]  # + OpenAI's tool call shape
                try: out = TOOLS[fn["name"]](**json.loads(fn["arguments"]))
                except Exception as e: out = f"ERROR: {type(e).__name__}: {e}"
                self.messages.append({"role": "tool", "content":
                    _truncate(str(out)), "tool_call_id": call["id"]})

if __name__ == "__main__":  # +
    agent = Agent(OpenAIModel())
    print(asyncio.run(agent.run(" ".join(sys.argv[1:]))))
    print(f"turns={agent.n_turns} cost=${agent.cost:.4f}", file=sys.stderr)
```

The `# /// script` header lets `uv run` install the OpenAI SDK, and the key comes from `OPENAI_API_KEY`. A simple task:

```bash
uv run harness.py "Create hello.py that prints hello world, \
run it, and tell me what it printed"
```

It wrote the file, ran it, and answered in two model calls, for about a hundredth of a cent:

```text
Created and ran `hello.py`. It printed:

hello world

turns=2 cost=$0.0001
```

The listing is adapted from [@barbasteHarnessEngineeringAnatomy2026], published under CC BY 4.0.

## Summary

- An agent harness is everything in an agent that isn't the model: the loop, tools, context and rules.
- Builders say to keep it thin and delete pieces as models improve.
- Research shows the harness still changes results, and that the best harness depends on the model.
- The parts that make up for a weak model move into skills, configuration, the model itself or other harnesses. Sandboxing, context management and evals stay.
