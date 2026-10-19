---
title: Hybrid Reasoning Agents
slug: hybrid-reasoning-agents
date: 2026-10-19 07:50
modified: 2026-10-20 09:36
status: draft
category: paper
summary: Agents that think in latent space and act through text.
cover: /_media/hybrid-reasoning-agents/hybrid-reasoning-agents-cover.jpg
hide_cover_in_article: true
tags:
  - ReasoningModels
  - AgenticReasoning
source: https://arxiv.org/abs/2610.18861v1
paper_title: Hybrid Reasoning Agents
paper_url: https://arxiv.org/abs/2610.18861v1
paper_authors: Jiaru Zou, Yunzhe Qi, Sirui Chen, Federico Bianchi, Shang Zhu, Pan Lu, Jingrui He, James Zou
paper_year: 2026
doi: 10.48550/arXiv.2610.18861
code: https://github.com/togethercomputer/HybridAgent
---

Two current approaches to [LLM Reasoning](llm-reasoning.md): either reasoning in token space or reasoning in latent space.

Latent reasoning is typically faster, since you don't have to decode every intermediate step into a token, but it lacks access to tools and the properties of transparency.

**Hybrid Reasoning Agents**, a paper from researchers at Stanford, Together AI and UIUC, proposes a "bridge" for these two directions, giving the model the ability to switch between latent reasoning and text-based interaction through a "unified generation process" [@zouHybridReasoningAgents2026].

Introduces a lightweight HybridLink, which enables transition between text and latent mode. Also a continuous working memory to preserve lossless hybrid context across turns.

Cool shit is that the latent thoughts can be inspected.

They also build post-training algorithms for hybrid reasoning: HybridSFT, tool-integrated hybrid generation that teaches the backbone to coordinate with HybridLink, and HybridRL, which scales reasoning by learning when to think in latent space and text.

Across 11 benchmarks, the 14B model beats the strongest agentic baseline by 17.4% on agentic tasks, while using 2.7x fewer text tokens and running 2.1x faster. The catch: against the best text-only RL baseline trained the same way, the gains are small (0.6 to 3.5 points), and frontier agents still beat it comfortably on Humanity's Last Exam.

## Introduction

For frontier intelligence, we need to operate not just on model reasoning, but on the world too. In other words, [Agentic Reasoning](agentic-reasoning.md) is the frontier of intelligence.

Text is how the model communicates with the environment.

A new paradigm of latent reasoning is being explored in parallel, which stops models needing to decode every intermediate step, which is more efficient.

The main issue is that i) they can't call tools and ii) they aren't interpretable, which means you can't keep tabs on them whilst training.

So the paper aims to combine the strengths of both.

This is a hard problem: it handles dense continuous vectors and sparse discrete tokens as hybrid inputs. Also determining when to transition.

### HybridLink

Lightweight residual adapter that bridges latent thoughts from the last-layer hidden space back into input embedding space. Enables sequence-wise recurrence within latent mode.

They use Qwen3-Base models [@yangQwen3TechnicalReport2025].

## Preliminary

Notation:

LLM agent $f_\theta$, using a standard Transformer with params $\theta$.

The agent autoregressively outputs in two modes: i) text and ii) latent.

Given an input context $x$ with input embeddings $E = [e_1, \ldots, e_t] \in \mathbb{R}^{t \times d_h}$, the agent computes the last-layer hidden state $h_t$ through a forward pass, and the hybrid input representation for the next step:

$$
e_{t+1} =
\begin{cases}
\mathrm{Emb}(y_{t+1}), \quad y_{t+1} \sim \mathrm{Softmax}(W_{\text{out}} h_t), & \text{if text mode,} \\
h_t, & \text{if latent mode,}
\end{cases}
$$

Where $W_{\text{out}}$ is the vocabulary projection matrix.

In text mode, the embedding is calculated from the softmax computation from the output hidden state of the network.

$\mathrm{Emb}(\cdot)$ is the token embedding function and $h_t$ is directly reused as the next latent input in standard latent reasoning.

The next hidden state of $f_\theta$ is then computed as:

$$
h_{t+1} = f_\theta([E_{\le t}; e_{t+1}])
$$

A complete single-turn hybrid reasoning trajectory $\tau$ can be represented as:

[![A single-turn hybrid reasoning trajectory. On the model side, a text chunk, a latent span, an ellipsis and another text chunk sit above the label Model-Side Hybrid Reasoning. On the environment side, a tool call leads through Tool Execution to a tool response, above the label Environment-Side Interaction.](../_media/hybrid-reasoning-agents/single-turn-trajectory.png)](../_media/hybrid-reasoning-agents/single-turn-trajectory.png)

*Equation 3 from [@zouHybridReasoningAgents2026], CC BY 4.0. Square brackets are text chunks and angle brackets are latent vectors.*

That sequence of text embeddings, latent vectors and tool call / response embeddings is the input for the next turn.

The text and latent outputs are model side, as are the tool calls. The tool responses are environment side. The multi-turn process stops when a final text answer is produced.

## Building a Hybrid Reasoning Agent

The lightweight adapter HybridLink supports latent and text modes in a shared interface.

HybridLink is a lightweight model, like 10M parameters.

It's computed as:

$$
e_{t+1} = \mathcal{R}(h_t) = h_t + W_2\,\sigma(W_1 h_t)
$$

where $W_1$ and $W_2$ are standard linear projections and $\sigma$ is an activation function like GELU [@hendrycksGaussianErrorLinear2016].

The residual connection largely preserves the original semantics of $h_t$, allowing the model to stay quite small.

In text mode, HybridAgent is just a standard LLM, sampling the next token from the vocab at each step.

The process continues until the agent generates one special token `<latent>`, which tells the model to enter latent mode.

Then for the next forward pass, the agent uses HybridLink for latent reasoning, in place of the LM head, and it keeps using it until it gets the exit anchor: `</latent>`.

To inspect any latent reasoning span, you can recover its semantics by mapping each latent step produced by HybridLink to its nearest token embedding.

It uses cosine similarity to find the token embedding closest to the latent step. This also allows us to decode the latent embeddings into tokens. Latent mode exits when:

$$
\arg\max_{v \in \mathcal{V}} \cos\left(\mathcal{R}(h_t),\, \mathrm{Emb}(v)\right) = \texttt{</latent>}
$$

[![A trajectory answering "Where is the company that Sachin Warrier worked for as a software engineer headquartered?" Step 1, text, 30 steps: "To solve this, I need to find out where the company that Sachin Warrier worked ... Let me first plan on the task step by step." Step 2, latent, 56 steps, read out as: "Step 1: Identify the company that Sachin Warrier worked for ... Step 2: Determine the location of that company ..." Step 3, a tool call: a search for "Sachin Warrier software engineer employment" returns "... software engineer in Tata Consultancy Services in Kochi. Later he resigned from the job ..." Step 7, latent, 49 steps: "From the search results, I can see that Tata Consultancy Services is headquartered in Mumbai, India. Therefore ..." Step 8, text, 10 steps: "The final answer is boxed Mumbai."](../_media/hybrid-reasoning-agents/hybrid-reasoning-agents-figure-2.png)](../_media/hybrid-reasoning-agents/hybrid-reasoning-agents-figure-2.png)

*Adapted from Figure 2 in [@zouHybridReasoningAgents2026], CC BY 4.0. The orange latent spans are read out by mapping each latent step to its nearest token embedding. "Steps" counts forward passes.*

In standard LLM agents, each turn's generated content and tool outputs are appended to the textual context, and re-encoded as the next turn's input prefix.

Hybrid trajectories interleave text tokens with latent vectors, and can't be constructed through conventional tokenization alone.

Instead, HybridAgent carries the complete context across turns in continuous working memory.

During input construction, all contents are mapped into the same input space through the input embedding layer, or HybridLink.

The shared space lets the Transformer process all 3 components uniformly and collect their KV entries throughout generation.

HybridAgent initialises working memory $\mathcal{M}$ from the input context $x$, then accumulates the KV entries generated throughout the first-turn trajectory $\tau^{(1)}$, i.e.,

$$
\mathcal{M}^{(0)} = \left(K_{\le |x|},\, V_{\le |x|}\right), \qquad
\mathcal{M}^{(1)} = \mathcal{M}^{(0)} \oplus \left(K_{\le t},\, V_{\le t}\right), \quad t = 1, \ldots, |\tau^{(1)}|
$$

## Learning to Perform Hybrid Reasoning

A nice property is that we don't need to pre-train from scratch, because the backbone architecture stays intact.

They adapt the standard agentic post-training pipeline of [Supervised Fine-Tuning](supervised-fine-tuning.md) and [Reinforcement Learning (RL)](reinforcement-learning.md) to the hybrid setting.

- SFT teaches the agent to coordinate with HybridLink and do reliable tool use.
- RL further scales reasoning by learning when to think in latent space and when to act through text.

For data, they filter existing text-based SFT datasets and convert them into hybrid trajectories through auto annotation. They assign reasoning steps to latent mode based on both LLM Experts' judgments and the target model's own prediction entropy.

### Format and HybridLink Activation

They adapt the backbone to the hybrid data format, and init the HybridLink.

They perform standard text SFT to teach the backbone when to enter and exit latent mode by emitting the two special anchor tokens.

They freeze the backbone and train HybridLink to align last-layer hidden states at latent positions with their input embeddings.

Activation objective:

$$
\mathcal{L}_{\text{Backbone}} = \mathrm{CE}\left(f_\theta(x),\, y\right),
$$

$$
\mathcal{L}_{\text{HybridLink}} = \lambda_{\cos}\left(1 - \cos\left(\mathcal{R}(h_{\text{lat}}),\, \mathrm{Emb}(y_{\text{lat}})\right)\right) + \lambda_{\text{mse}} \left\lVert \mathcal{R}(h_{\text{lat}}) - \mathrm{Emb}(y_{\text{lat}}) \right\rVert_2^2
$$

### Self-Conditioned HybridSFT

The activation provides a well-aligned initialisation, but the backbone has not learned to reason over latent thoughts produced by HybridLink.

So they do this "two-pass self-conditioned training":

- Pass 1: HybridAgent is teacher-forced on the ground-truth hybrid sequence.
- Pass 2: HybridAgent processes the same sequence but replaces the ground-truth latent embeddings with its own latent inputs.

They jointly supervise text and latent positions in this second pass as:

$$
\mathcal{L}_{\text{HybridSFT}} = \mathcal{L}_{\text{Backbone}} + \mathcal{L}_{\text{HybridLink}}
$$

### Scale Reasoning via Hybrid Reinforcement Learning

HybridAgent cannot straightforwardly adopt popular RL algorithms such as [Proximal Policy Optimization](proximal-policy-optimization.md) [@schulmanProximalPolicyOptimization2017] or [Group Relative Policy Optimisation](group-relative-policy-optimisation.md) [@shaoDeepSeekMathPushingLimits2024], as they assume every generated position carries a token-level likelihood.

They formalise Hybrid Policy Factorization.

The importance ratio from [Importance Sampling](importance-sampling.md) is undefined because HybridAgent generates continuous vectors, not sampled tokens. Policy sampling is confined to model-generated text positions only.

And introduce Policy Update via Exact Rollout Replay.

## Evaluation

They test on 11 benchmarks in three groups:

- Maths: AIME 25, AIME 26 and MATH500.
- Knowledge-intensive: GPQA-Diamond, MedQA and MuSiQue.
- Agentic: GAIA, Humanity's Last Exam (HLE), WebWalker, SealQA and DeepSearchQA.

They compare against four kinds of baseline: text-based agentic training (standard text SFT, GRPO, ARPO), latent reasoning methods (Coconut [@haoTrainingLargeLanguage2026], SoftThinking, SoftCoT), frontier LLM agents (GPT-5.6-Luna, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4.1-Flash, QwQ-32B) and agentic frameworks (ReAct, Search-o1, WebThinker).

The backbones are Qwen3-Base at 4B, 8B and 14B. HybridSFT uses 54K hybrid trajectories from their annotation pipeline, then HybridRL uses 12K deep reasoning examples and 1K agentic search examples. The agent gets two tools, a sandboxed Python interpreter and a live search engine, capped at 10 calls per trajectory.

No current serving engine supports multi-turn hybrid generation, so they had to extend the training and inference infrastructure themselves to handle the mode switches and carry the KV working memory between turns.

### Results

On the agentic benchmarks, HybridAgent (14B) beats the strongest agentic baseline by 17.4% on average. On GAIA it averages 52.4, versus 43.7 for ARPO at the same size and 41.7 for Kimi-K3. On WebWalker it gets 49.5, versus 40.5 for ARPO (Table 1).

It doesn't win everywhere. On HLE the frontier agents are well ahead: Kimi-K3 averages 31.4 and GPT-5.6-Luna 29.0, against 16.0 for HybridAgent (14B). So the Table 1 caption's claim that it "consistently outperforms advanced LLM agents" doesn't hold there. GPT-5.6-Luna also edges it on DeepSearchQA F1 (49.4 vs 46.8).

The pure latent baselines fall apart on agentic tasks (Coconut gets 6.8 on GAIA at 14B), which makes sense, since they can't call tools at all.

Under matched model sizes and training budgets, they report gains of 18.5% at the SFT stage and 10.4% at the RL stage. Those averages are over both the text and the latent baselines, including weak ones like Coconut. Against the best text baseline the margins are much smaller. After SFT, HybridAgent is roughly level with standard text SFT. After RL it beats the best of GRPO and ARPO by 0.6 to 3.5 points on the maths and knowledge benchmarks (e.g. AIME 26: 45.8 vs 43.3 at 14B, Table 2).

For [Test-Time Scaling](test-time-scaling.md), they measure pass@k (k from 4 to 64) on 100 hard maths and science questions. HybridAgent gets more out of extra samples than text agents trained with GRPO (Figure 5).

### Efficiency

- 2.2x to 3.1x fewer text tokens across model scales (2.7x on average).
- Up to 54% fewer total generation steps.
- 2.1x faster inference.
- 2.7x to 3.7x lower compute cost (Appendix G.2).

Each latent step is still a full forward pass through the model. Latent mode only skips the LM head, sampling and token embedding lookup. So the speed-up appears to come mostly from taking fewer generation steps overall, rather than from each step being cheaper.

## In-depth Analyses

### Training dynamics

Self-conditioning is what makes HybridSFT work. On the 14B backbone (Table 3), MATH500 goes from 47.0 for the base model, to 62.4 after format activation, 73.8 after HybridLink activation, then 84.2 with full self-conditioned HybridSFT. That matches standard text SFT (83.8). Coconut gets 61.2. Self-conditioning adds 30.5% on average over HybridLink activation alone.

It also makes the latents more readable. Over 200 questions, the latent readouts get more readable and more context-specific with each stage (Figure 8), while the weaker methods produce "largely unreadable artifacts of the embedding geometry".

During HybridRL, the 14B agent moves more and more of its generation into latent mode, while reward and tool use both go up (Figure 9). There's no reward for using latent mode, so it learns this by itself.

Reward ablation (8B, Table 4): removing $r_{\text{correct}}$ hurts most (MATH500 84.6 to 79.2, GAIA 43.7 to 35.9). Removing $r_{\text{tool}}$ barely matters on MATH500 (84.2), where Python is usually enough, but drops GAIA to 38.8. The tool reward only goes to trajectories that use both Python and search.

### Does it actually use the latent thoughts?

They mess with the latent vectors on MATH500 and see what breaks (Table 5):

| Intervention | 8B | 14B |
|---|---|---|
| None | 84.6 | 88.2 |
| Replace with nearest token embedding | 82.6 | 87.2 |
| Block latent mode (text only) | 80.4 | 78.8 |
| Shuffle latents within a span | 73.2 | 76.4 |
| Gaussian noise ($\sigma = 0.5$) | 72.6 | 75.0 |
| One latent step per span | 72.2 | 75.0 |
| Gaussian noise ($\sigma = 2.0$) | 66.4 | 70.2 |
| Replace with the mean latent | 53.6 | 58.4 |
| Swap in latents from a different problem | 44.8 | 48.6 |
| Replace with a random token embedding | 41.2 | 44.6 |

Swapping in latents from a different problem hurts more than heavy Gaussian noise, even though it's a smaller perturbation (relative size 1.4 vs 2.0). So it's the content of the latent thoughts that matters, not just how much you disturb them. Order and depth matter too.

The second row is the interesting one. Replacing every latent with its nearest token embedding only costs 1 to 2 points. Most of what a latent step carries survives being snapped to a single token, which fits with the readouts being so readable. The results suggest the advantage over text is more about efficiency than about latents carrying much extra information.

### HybridLink design

They compare HybridLink designs on the 14B backbone (Table 6):

| Design | MATH500 | MedQA | MuSiQue |
|---|---|---|---|
| Identity (feed $h_t$ straight back) | 71.4 | 73.7 | 24.5 |
| 1-layer | 83.6 | 79.7 | 28.5 |
| 1-layer + residual | 86.4 | 80.3 | 31.0 |
| 2-layer | 85.8 | 81.0 | 31.5 |
| 2-layer + residual (HybridLink) | 88.2 | 81.7 | 34.5 |

Feeding $h_t$ straight back, the way standard latent reasoning does, is by far the worst. The residual connection adds 4.3% on average.

### Tool use and auditability

Compared with text agents trained with GRPO, HybridAgent makes valid tool calls more often at both 8B and 14B, with the same 99.7% execution success. Accuracy on questions where it uses tools is up 2.2% to 44.1%, and the useful tool-call rate (calls whose results contain the final answer) is up 4.5% to 7.7% (Figure 10).

For auditability, they check whether the text that's left is still enough for an outside observer, with the latent spans hidden. 560 problems from MATH500 and AIME 25/26 (Table 7):

- **Visible trace:** 43% to 49% fewer text tokens (284 vs 497 at 14B), plus 170 to 183 latent steps per problem, but 26% to 28% more key reasoning steps per 100 visible tokens.
- **Monitoring:** a monitor predicting correctness from the visible trace stays within 0.022 AUROC of the text agents.
- **Debugging:** a judge finds the first error in 98% to 99% of failed rollouts, with under 1% answer leakage in its hints.
- **Correction:** targeted hints improve the fix rate over a generic retry by 9.1 (8B) and 5.4 (14B) points, versus 3.4 and 3.5 for the text agents.

Because the latent spans were hidden, these tests show the visible text is still enough to monitor and debug the agent. They don't show that the latent readouts are a faithful account of what the model computed. Figure 8 only measures how readable and context-specific the readouts are.

## Discussion

Safety angle: frontier labs hide the chain of thought from API customers, but a recent paper showed the encrypted traces returned to clients can be replayed into a sibling model and transcribed word for word [@panfilovStealingReasoningTraces2026]. With HybridAgent, the latent thoughts are only readable through the provider's own HybridLink, so there's no text to leak, and the provider can still monitor the full readout.

Future directions:

- An agent harness where KV working memory is first class: context management that works directly on KV entries, workflows that schedule steps by mode (tool calls and human checkpoints in text, routine computation in latent), and agents passing hybrid trajectories to each other as messages.
- Cheaper post-training: since the latents are readable during rollout, a critic could give step-level credit to latent spans, or kill a trajectory whose latent reasoning has already gone wrong before any text or tool call.

## Takeaway

HybridAgent thinks in latent space and acts through text in a single generation. You can build it from any pre-trained LLM with a tiny adapter plus HybridSFT and HybridRL.

- **Useful:** a recipe for adding latent reasoning to an agent without giving up tool calls or a readable trace.
- **What the evidence supports:** big efficiency gains (2.7x fewer text tokens, 2.1x faster), modest accuracy gains over well-trained text agents, and an agent that really does rely on its latent thoughts.
- **Boundary:** the models stop at 14B, frontier agents still lead on the hardest benchmark (HLE), and the latent readouts haven't been shown to be faithful.
