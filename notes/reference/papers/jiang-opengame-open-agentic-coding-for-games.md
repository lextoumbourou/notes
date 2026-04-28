---
title: "OpenGame: Open Agentic Coding for Games"
date: 2026-04-29 00:00
modified: 2026-04-29 00:00
status: draft
category: reference/papers
tags:
- AgenticReasoning
- GameDevelopment
---

*My notes on paper [OpenGame: Open Agentic Coding for Games](https://arxiv.org/abs/2604.18394v1) by Yilei Jiang, Jinyuan Hu, Qianyin Xiao, Yaozhi Zheng, Ruize Ma, Kaituo Feng, Jiaming Han, Tianshuo Peng, Kaixuan Fan, Manyuan Zhang, Xiangyu Yue

A paper very relevant to my interests right now: **OpenGame: Open Agentic Coding for Games** [@jiangOpenGameOpenAgentic2026], describes **OpenGame** an agentic framework designed for end-to-end web game creation.

The paper argues that to build products as complex as games, the field needs to move beyond *generalist code agents* to *specialist frameworks*. Reminds me of the [SheetCopilot Agent](../../permanent/sheetcopilot-agent.md), an agentic-framework for Spreadsheet controls and even systems like [AlphaEvolve](alphaevolve-a-coding-agent-for-scientific-and-algorithmic-discovery.md), a system for algorithmic discovery.

The paper basically throws the kitchen sink at the problem of game design, including building a base model, a code agent, a new collection of [Agent Skills](../../permanent/agent-skills.md) for game development, and a new benchmark and evaluation framework.

![jiang-et-all-figure-2.png](../../../../_media/jiang-et-all-figure-2.png)

### Base Model

They build a new foundation model based on [Qwen3.5-27B](Qwen3.5-27B.md) backbone called GameCoder-27B.

[Continual Pre-Training (CPT)](../../../../permanent/continual-pre-training-cpt.md): on a corpus from open-source Phaser and JavaScript/Typescript game repositories on GitHub, with docs and tutorials to build strong prior over game loops, physics systems, asset usage, state management etc.
[Supervised Fine-Tuning](../../permanent/fine-tuning.md) on game generation prompts using `gpt-codex5.1`, with solutions from `minimax2.5`. E.g. `"Implement a 2D platformer character controller with double-jump and sprite animations"`
[Reinforcement Learning (RL)](../../permanent/reinforcement-learning.md) with execution and feedback at components level. Get it to make single-file gameplay logic and targeted functional modules (e.g., collision detection, state-machine transitions), then test the code with unit tests and check execution success and aggregate test pass rate. Getting the model to be strong at the component level works, because they use a downstream agent to assemble building blocks into a full multi-file project.

### Code Agent Design

To produce a complete game you need [Structured Long-Horizon Workflows](Structured%20Long-Horizon%20Workflows.md).

OpenGame orchestrates the agent through six operational phases:
- Initialisation and Classification
- Scaffolding and Design Generation
- Asset Synthesis
- Code Implemenation
- Verification
Uses a persistence todo_write tool that allows agent to plan, exectture, transition across these phasse sin a controlled manager.

#### Initialisation and Classification

establishing a macro-level execution plan.

Agent first invokes classify-game-type tool.

Rather than relying on ambiguous genre labels, this tool applies a Physics-First Classification rule that categorizes the task according to physical constraints and spatial mechanics (e.g., mapping “falling without ground support” to a platformer archetype or “snapping to a grid” to grid_logic)

#### Scaffolding and Design Generation

Now that we know the game archetype

agent executes a scaffolding procedure through run_shell_command

operation copies the shared core, the appropriate modules/{archetype} codebase, and the relevant architectural documentation (docs/ ) into the workspace, which creates a stable structural baseline before game-specific implementation begins.

agent then invokes generate-gdd to produce a technical Game Design Document (GDD).

This tool dynamically loads archetype-specific API constraints from the scaffolded documentation, ensuring that the proposed mechanics remain feasible under the selected framework. The agent extracts the implementation roadmap from the GDD and uses todo_write to refine its high-level plan into granular, file-specific actions.

#### Multimodal Asset Synthesis

asset phase, the agent first reads asset_protocol.md through read_file to ensure parameter compliance. It then invokes generate-game-assets, leveraging multimodal generation models to synthesize backgrounds, character animations, static items, and audio assets from the GDD’s asset registry. For tile-based games, generate-tilemap converts ASCII layouts into structured JSON tilemaps. Finally, by reading the produced assetpack.json, the agent records the exact texture and asset keys required during implementation, substantially reducing downstream asset-reference hallucinations



- **Game Skill** is a "reusable, evolving capability" composed of a Template Skill that grows a library of project skeletons from experience, and a Debug Skill that maintains a living protocol of verified fixes-together wenabling the agent to scaffold stable architectures and repare integration errors, rather than patch isolated bugs.

They also introduce a model **GameCoder-27B** a code LLM specialised in game engines, created through 3-stage pipeline contrinual pre-training, supervised fine-tuning adn execution-grounded reinforcement learning.

Also introduces **OpenGame-Bench**, evaluation pipeline that scores agentic game generation that measures:
Build Health
Visual Usability
Intent Alignment via headless browser execution and VLM judging.


---

### Other notes

There are 3 common failure models:
- Logical Inchoerence: model loses track of global state across game loop, causing freezing, failures to terminate or never realized key mechanics.
- Engine-Specific Knowledge Gaps: general models often ignore or misuse engine abstractions, re-implememnting mechanics from scratch instead of correctly leveraging framework-native physics, scene and event systems.
- Cross-File Inconsisences: individual files are plauseible, but oberall poreject breaks due to mistmatches assets keys, flawed scene wiring, mising config fields or broken init order.

They argue that field must move beyond "generalist code agents" to "specialist frameworks" that understand the intrinsic structure of games

Template Skill grows an evolcing library of specialise project skeletons (L) starting from a game-agnostic meta template ($M_0$) and explanding intor specliased template familiyes live gravity-based side view and top-down continuous motion.

This sharply reduces the search space of generation and stabilises project-wide structure.

**Debug Skill** maintains a livign debugging protocol (P) updated from observed build, test and runtime outcomes - it's a living debugging protocol which is updated from observations ocross build, test, runtime outcomes. Letting the agent accumulate verified fixes and systematically resolve high-frequency integration failures.

---


#### Context-Aware Code Implementation

Befoer writeing gameplay logic, the agent merges GDD parameters into `gameConfig.json`, enforcing data-driven interface between design and code.

To mitigate context overfows during implementation, we introduce a Three-Layer Reading Strategy:
- Using `read_file`, the agent progressively loads:
    - (1) an API summary for the template system
    - (2) the targetted source file (_Template*.ts) that will be modified
    - (3) the implemntation guide, loaded last to maximumse immeditate salience.
- Code generation follows a Template Method Pattern: rather than than writing the proejct from scratch, teh agent copies template files and overrides designated hook methods (eg. setupCustomCollisions) to inject game-specific logic while preserving the determiniisti lifecycem maangemnet of base clases.

### Verification and Self-Correction

final phase: the agent enters a verification and self-correction loop

reads debug_protocol.md to perform a static self-review over common generative failure modes
Uses run_shell_command to execute npm run build and npm run test under headless browser evaluation

When build or test failures occur, the agent parses compiler output, localizes the faulty script, and iteratively repairs the project until a playable game is obtained. This protocol provides the operational substrate for the more general Debug Skill described next.

### 3.3 Agent Evolution with Game Skills

Game Skill, a reusable capability for converting a natural-language game specification into a runnable project. Game Skill consists of two components: Template Skill, which stabilizes project structure, and Debug Skill, which improves reliability during verification and repair.

Problem setting. Given a user specification x describing mechanics, theme, and constraints, the agent must produce a project y that can be built and executed. In practice, failures are more often caused by cross-file inconsistenciesspanning assets, configuration, scene wiring, and initialization order—than by isolated syntax errors. Game Skill is designed to reduce these systemic failures while keeping generation stable across diverse requests.

Template Skill. The agent begins with a single meta template M0, a minimal game-agnostic project skeleton that defines the universal structure required for a playable game, including project layout, initialization, asset loading,  5 OpenGame: Open Agentic Coding for Games  Algorithm 1: Game Skill execution  Input: User specification x, meta template M0, template library L, debug protocol P Output: Runnable game project y Select a template family T ∈ L (initialized as M0 at the beginning of training); Instantiate T to scaffold a project skeleton y; Generate game-specific content conditioned on x within the extension points of y; repeat// until convergence  Run verification and execution (build, test, run) guided by P; if failure observed then  Diagnose the failure using P and repair y; Append a verified (signature, cause, fix) entry to P if the pattern is new;  until y is buildable and runnable;  Optionally extract reusable fragments from y and merge into L; return y  scene loops, and configuration interfaces. M0 intentionally does not assume any genre, physics regime, or gameplay mechanic.