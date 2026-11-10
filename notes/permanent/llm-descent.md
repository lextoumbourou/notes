---
title: "Weight-Space Programming: training neural networks like a 1990s researcher"
slug: llm-descent
date: 2026-11-07 00:00
modified: 2026-11-10 06:10
status: draft
summary: "What an LLM actually does when it writes a neural network's weights."
cover: /_media/llm-descent-cover.jpg
cover_credits: Photo by <a href="https://unsplash.com/@adiyusuf90">Adi Yusuf</a> on <a href="https://unsplash.com/photos/people-hiking-on-foggy-mountain-j-UTH7jAnUs">Unsplash</a>
category: essay
tags:
  - LLMs
  - Interpretability
---

## Intro

[Weight-Space Programming](weight-space-programming.md) still has me pretty intrigued this week. The basic idea is that frontier LLMs can read, write and edit neural network weights, though there's certainly nuance to what's going on here. Still, it feels like there's a lot of new ground to explore.

In the [original blog post](https://rikiyat.github.io/blog/weight-space-programming/), beyond the reading and editing experiments, the authors had Astra build a network to play Go from scratch, then improve it by playing practice games, analysing how it lost, and editing the weights. It really seems like a new training paradigm, something like *LLM Descent*, but is it?

In this article, I run my own weight-writing experiment with Claude Opus 5.5 on a familiar image classification problem, and compare what it does with some well-known but largely forgotten pre-deep-learning techniques. Then I look more closely at what Astra was actually doing when it built its Go network from scratch, and find that it looks very similar.

## Building an image classifier with LLM Descent

The MNIST digit dataset is the OG ML image classification problem. If you've come across this blog post, chances are you're familiar with it.

[![Twenty handwritten digits from the MNIST training set, two of each digit from 0 to 9, drawn in dark ink on off-white 28 by 28 pixel squares.](../_media/llm-descent/mnist-examples.png)](../_media/llm-descent/mnist-examples.png)

I built a test with a fixed CNN, then instructed Claude to write code that sets all its weights, with a few rules: no example digits, and no fitting, regression, or searching. That's a tighter setup than Astra had for Go: no scipy or PyTorch and no practice games, only a score on the training set after each attempt.

Claude has to set all 5,994 weights of this network:

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 5),    # conv1: 1x28x28 -> 8x24x24
    nn.ReLU(),
    nn.MaxPool2d(2),       #        8x24x24 -> 8x12x12
    nn.Conv2d(8, 16, 5),   # conv2: 8x12x12 -> 16x8x8
    nn.ReLU(),
    nn.MaxPool2d(2),       #        16x8x8 -> 16x4x4
    nn.Flatten(),          #        16x4x4 -> 256
    nn.Linear(256, 10),    # fc:    256 -> 10 class logits
)
```

You can read [the whole prompt](https://github.com/lextoumbourou/weight-space-programming-experiments/blob/main/mnist-training/results/opus-p8-3/prompt.md) on GitHub.

Claude's code runs in a sandbox that can only import numpy and math. Once it returns the weights, a test harness runs the network in MLX on the training and test images.

Each step, Claude gets all the feedback so far and has up to 50 steps to solve the problem. (The harness calls these 10,000 images the validation set, which is why the feedback says "validation", but since Claude learns from them they're really its training set.) Runs stop early once the training loss stops improving, and the network with the lowest training loss is the one scored on the test set.

The training set here is 10,000 images held out from MNIST's own training split. The official 10,000 MNIST test images are scored at every step but never shown to Claude.

[![A loop of four flat coloured blocks joined by arrows: Claude writes build_weights(); a sandbox sets all 5,994 weights; the network runs on 10,000 training images; Claude gets back the loss, accuracy and confusion matrix and writes the next step. A separate block notes that the test set is never shown.](../_media/llm-descent/loop.png)](../_media/llm-descent/loop.png)

Example of the feedback it gets each step:

```text
Step 1 of 50: validation loss 1.8708, accuracy 0.4273.

Confusion matrix (rows: true digit, columns: predicted digit):

true \ predicted     0     1     2     3     4     5     6     7     8     9
               0   539    70   296     0     0    12    29    45     0     0
               1     0  1049     0     0     0     0     0    15     0     0
               2     4   296   219   182     0     6    14   269     0     0
               3     0   307     9   405     4    32    13   260     0     0
               4     0   546     6     1   338     0    11    81     0     0
               5     0   486     8    35    18   281    43    43     0     1
               6     5   430    17     0     2    51   456     6     0     0
               7     0   189     6     0    23     1     0   871     0     0
               8    12   460    26    83     4    12    48   361     0     3
               9    27   319    32     3    61     7     2   395     0   115

History (validation set, 10,000 images):

step | loss | accuracy
---- | ---- | --------
1 | 1.8708 | 0.4273

At most 49 steps left. Reply with the complete build_weights() for step 2 in one Python code block.
```

### The descent

I ran it three times.

[![Step chart of the best training loss so far for three Opus 5.5 runs, with each step's loss as faint dots and probe steps as ticks along the top. Run 1 flattens at 1.25 (65% test accuracy) and stops at step 17. Run 2 descends steadily to 0.86 (72%) and stops at step 26. Run 3 keeps descending, with many probe steps, to 0.42 (86%) at the 50-step cap.](../_media/llm-descent/loss-by-step.png)](../_media/llm-descent/loss-by-step.png)

All three runs descend, but they land in very different places: 65%, 72% and 86% on the test set. The difference comes down to the approach each run takes, in particular a trick that two of the three runs invented and run 3 used over and over.

### A 1990s computer-vision pipeline

Here's run 3, step by step. Every stage is a technique from before deep learning. In 1998, the paper that introduced MNIST described that era's approach: "Traditional pattern recognition is performed with two modules: a fixed feature extractor and a trainable classifier", and the feature extractor "is often entirely hand crafted" [@lecunGradientbasedLearningApplied1998]. That's the shape of everything Claude builds below.

[![Line chart of test accuracy for each classifier Opus 5.5 wrote in run 3, over 50 steps, on four coloured bands. Hand-drawn templates (steps 1 to 13) climb from 43% and stall at about 67%. First probes (steps 14 to 20) are marked as ticks at the bottom. Naive Bayes from measured features (steps 21 to 42) lifts accuracy to about 85%. Calibration (steps 43 to 50) ends at 86%. Adam after 50 gradient steps is marked at 92%.](../_media/llm-descent/run-3-story.png)](../_media/llm-descent/run-3-story.png)

#### 1. Hand-made features and templates (steps 1 to 13)

Claude starts like a computer vision engineer from the 1990s. Its first layer is a set of oriented line detectors and an "ink present" blur, written from formulas, much like the Sobel edge gradients that 1990s digit recognisers started from [@favataMultipleFeatureResolution1996]:

[![Two rows of eight 5 by 5 first-layer filters, teal for positive and pink for negative. Opus 5.5's row: four clean line detectors at 0, 45, 90 and 135 degrees, then four soft blob-shaped detectors. Adam's row: irregular, mixed patterns.](../_media/llm-descent/filters-written-vs-learned.png)](../_media/llm-descent/filters-written-vs-learned.png)

On top of those, its second layer detects shapes: loops, horizontal and vertical bars, and "cavities" open to the left, right, top or bottom, the structural and concavity features that handwriting recognisers relied on [@trierFeatureExtractionMethods1996] [@favataMultipleFeatureResolution1996]. Then it draws a stroke template for each digit and classifies by template matching, the oldest approach in character recognition [@moriHistoricalReviewOCR1992], nudging the details using the confusion matrix. That gets it from 43% to about 67%, and then it stalls:

> Accuracy is stuck at about 66%, and the results swing wildly with κ, the guessed ratio of real feature strength to template strength.

#### 2. Measuring without looking (steps 14 to 20)

A 1990s engineer would have looked at the digits to see where their features go wrong. Claude can't. At step 14, it does something I didn't expect:

> My mental model of where the loop detectors fire must be wrong, and I have no direct view of the features. So this step spends one evaluation on measuring them.

It stops trying to classify digits. Instead, it rewires the output layer so each of the 10 outputs means a combination of its own features, and uses the confusion matrix to read back how often each feature fires in each digit.

[![Two heatmaps. Left, what came back: the step 22 confusion matrix, rows true digit and columns the eight feature combinations the probe network could output (none, top bar, bottom bar, both, upper-left vertical, and so on). Right, what it learned: the decoded share of each digit with a top bar, a bottom bar and an upper-left vertical stroke. For example 97% of 7s have a top bar and 1% of 1s do.](../_media/llm-descent/probe.png)](../_media/llm-descent/probe.png)

A probe step is useless as a classifier (it scores between 3% and 24%), but it costs nothing: only the network with the lowest training loss gets scored.

Is that cheating? It doesn't break the rules: the code never sees an image, never runs the network and never fits anything. But it squeezes far more out of the training set than the loss alone: class-by-class feature statistics, read through a 100-number window one step at a time. The test set stays untouched, so the scores below are still honest held-out scores.

#### 3. Naive Bayes (steps 21 to 42)

With a table of measured feature probabilities per digit, Claude reaches for a classic:

> I now have per-digit probabilities for about 18 topology features from the probes. That is exactly what naive Bayes needs.

Accuracy jumps from 71% to 76%. From there it alternates: probe a new feature, add it to the naive Bayes, check the loss. By step 41, after 13 probes, it's at 86%.

> Features with measured per-digit statistics clearly beat hand-guessed templates.

By the end, the final layer is textbook naive Bayes for yes/no features, sitting on top of the original templates. Bayesian decision rules for reading characters go back to 1957 [@chowOptimumCharacterRecognition1957]. About 35 features each add log(p / (1 − p)) to a digit's score when they fire, with every p a number it measured through a probe and typed into its code (a top bar fires in 990 of every 1,000 sevens, and 7 of every 1,000 ones). It even scales the naive Bayes terms down, because its features overlap and naive Bayes overcounts overlapping evidence.

#### 4. Calibration (steps 43 to 50)

The last few steps are fine-tuning by hand: sharpening or softening the logits (temperature scaling) and correcting class priors from the column totals.

> The loss is flat between logit scales of 1.1 and 1.2, so the optimum is close to 1.15.

[![Three training confusion matrices for run 3, rows true digit and columns predicted digit, shaded pink. At step 1 (43% training accuracy) most digits are predicted as 1 or 7. At step 13 (66%) the diagonal is visible but noisy. At step 50 (87%) the diagonal dominates.](../_media/llm-descent/confusion-over-steps.png)](../_media/llm-descent/confusion-over-steps.png)

#### When fitting was allowed, it reached for 1936

My first version of the prompt forgot to ban fitting. By step 4, Opus had written this:

```python
for k in range(300):                      # 300 random synthetic digits per class
    X[dgt, k] = features(place(polys, slant=rng.uniform(-2, 5), ...))
W = np.linalg.solve(Sig, (mu - mbar).T).T  # a linear discriminant fitted to them
```

It drew 300 random versions of each digit from its own stroke models and fitted a linear discriminant to them, Fisher's technique from 1936: training, on data it made up. That run reached 83% by step 14, before I stopped it and tightened the rules.

### Where it landed

[![Bar chart of test accuracy. Random search with 50 evaluations: 16%. Opus 5.5 run 1: 65%. Run 2: 72%. Run 3: 86%. Opus 5.5 with fitting allowed, 14 steps: 83%. Adam after 50 gradient steps: 92%. Adam after 5,000 steps: 99%.](../_media/llm-descent/final-scores.png)](../_media/llm-descent/final-scores.png)

- **Random search** gets the same feedback but no reasoning: 16% after 50 tries.
- **Opus 5.5** lands between 65% and 86%, depending mostly on whether it discovers probing. Run 2 never probed and flattened at 72%; run 1 started probing at step 13 and the stopping rule ended it mid-measurement.
- **Adam** still wins easily: 92% after 50 gradient steps, 99% after 5,000. LLM Descent is not an efficient optimiser.
- The three runs cost $2.56, $5.93 and $9.44 in tokens.

## Exploring the Go network

Claude's MNIST network turned out to be a 1990s computer-vision pipeline. Does the paper's own Go network work the same way?

To me, the from-scratch Go network was the most striking result in the original post.

Astra was given a 1.5M-parameter 9x9 Go policy network (243 inputs, six hidden layers of 512 units, and 82 outputs, one per point plus pass) with every weight set to zero. It had a Python sandbox, but no training, fitting or search of any kind, and it was asked to make the network play Go as well as possible. In most runs it could also spend "passes": each pass plays up to 128 practice games and shows it the results. Runs got 0, 10, 25 or 50 passes.

With 50 passes, its network won 84% of games against one small PPO network and 81% against another, neither of which it had played during development. It also beat a PPO network of the same size, trained on 20.5 million games, in 70% to 79% of games. With no passes at all, a single 336-line Python run still won just over half its games against the smaller PPO networks [@takehi2026weightspaceLogs].

So, how is this even possible? Well, Astra spells it out in its [final answer](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/runs/budget_50/final_answer.md):

> The hand-coded, symmetry-preserving policy combines approximate group-liberty propagation, captures, connected-group rescues, self-atari and eye avoidance, and ko/snapback risk suppression.

I don't even know how to play Go, so I'm not going to pretend I understand this, but my LLM tells me that every item on that list is a staple of classical Go programs: counting liberties to tell which groups can be captured, recognising eyes to tell which groups live, handling ko, and estimating territory with an influence function, where each stone radiates influence that fades with distance. The first Go programs were built entirely on influence functions [@bouzyComputerGoAI2001] [@mullerComputerGo2002].

Behind those 1.48 million weights is a [config of 26 hand-set numbers](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/runs/budget_50/work/final_policy_config.json), compiled into the network by a design script:

```json
{
  "capture": 9.0,
  "rescue": 8.0,
  "self_atari": 7.5,
  "eye": 11.0,
  "contact": 3.0,
  "own_adj": 0.8,
  ...
}
```

The design script builds the network layer by layer, like a hand-written board analysis: layer 1 counts each stone's liberties, layers 2 and 3 spread that evidence through connected stones, layer 4 flags groups in atari and self-atari, layer 5 scores captures and rescues, and layer 6 combines tactics with connection, territory and eyes. The 26 numbers mostly set how much each rule counts.

Its improvement from practice games was tuning those numbers by hand: between its first design and its final network, `rescue` went from 6.0 to 8.0, `contact` from 1.4 to 3.0 and `own_adj` from 0.25 to 0.8. Practice games also led it to add five new rules along the way, such as a penalty for ko repetition (`repeat_risk`) and a bonus for third-line frameworks (`framework_bonus`). Play, diagnose, add a rule or retune one: that's how rule-based Go programs were built for decades [@mullerComputerGo2002].

So the paper's most impressive from-scratch network is a rule-based Go program, compiled into weights. None of this was hidden - the authors' rules explicitly allowed writing weights from its own design, and Astra calls its own policy hand-coded. A hand-written program that beats a PPO network trained on 20.5 million games is still impressive engineering. It doesn't look like a new kind of training so much as a clever combination of training-free tricks of yesteryear.

## What this training actually is

Every step Claude took on MNIST, and every part of Astra's Go program, is a technique from the classical toolbox:

| What the model did | Classical technique | Around since |
|---|---|---|
| Oriented line filters in the first layer | edge detection (Sobel and friends) | 1960s |
| A stroke template per digit | template matching | 1950s |
| Loop and cavity detectors | structural features for handwriting recognition | 1960s to 1990s |
| Per-digit feature probabilities, summed as log-odds | naive Bayes | 1960s |
| Logit scale and class-prior corrections | probability calibration | 1990s |
| One setting at a time, split the difference (Claude's calibration, Astra's Go coefficients) | coordinate descent and line search | long before computers |
| Liberties, eyes and influence (Astra) | rule-based Go evaluation | 1970s |
| Synthetic digits plus a linear discriminant (the first attempt) | linear discriminant analysis | 1936 |

The network is never used for what makes neural networks special: learning features nobody wrote down. It's a compilation target. Claude designs a classical algorithm, then translates it into weights.

Astra's Go network, as we saw, is the same thing at a bigger scale.

So, at least for writing weights from scratch, weight-space programming is ordinary programming with the weights as the compile target, much like DeepMind's [Tracr](https://arxiv.org/abs/2301.05062), which compiles programs into transformer weights [@lindnerTracrCompiledTransformers2023]. The model writes a program in its head and delivers it as a network. None of this tests the other half of Weight-Space Programming, reading and editing weights that gradient descent trained, which is where the paper's headline results are.

## LLM Descent: is it actually a thing?

<!-- LEX: your section. Points the results support either way:
- For: the training loss goes down on every run, step after step, using nothing but feedback (the loss-by-step chart). Each step is an update to the network driven by a measured signal, which is what training is.
- Against: no step is a gradient. Each one is a hypothesis and a test: change one setting and keep or revert it (coordinate descent, line search), or deliberately make the loss worse to measure something (probes). What descends is the hand-written program's handful of parameters, not the network's weights.
- Efficiency: it beats random search with the same 50 evaluations by a wide margin, so the reasoning is doing real work per step. But each step costs cents to dollars, and Adam needs only 50 to 100 cheap steps to match or beat it.
- Probing is the most "non-descent" part: steps that are useless as classifiers, worth it only because the best step is the one scored.
- Possible verdict: "LLM Descent" is a good name for the loop, but what it optimises is a classical program, so it's closer to an engineer tuning a system than to gradient descent. -->

## The ceiling is the old ceiling

Claude topped out at 86% on MNIST, while Adam reaches 92% after 50 gradient steps and 99% after 5,000. That's the ceiling hand-crafted features always had: Claude can only use features it can describe.

## What's new

<!-- LEX: in your words. Candidates: (1) it measured the data without ever seeing it, rebuilding the "look at the data" step out of a loss and a confusion matrix; (2) when its hand-written features stalled, it switched from guessing to measuring; (3) it did a feature engineer's day of work for between $2.56 and $9.44. -->

## Next

<!-- LEX: other models (an earlier 15-step test suggested a big jump from Opus 4.x to Opus 5.5), other data (audio), editing a trained network, and a follow-up on tabular data (Titanic, an anonymised Titanic that Claude saw through, and synthetic data with a known answer): what Claude does when it already knows the answer. -->

## Code and results

Everything is in the [experiments repo](https://github.com/lextoumbourou/weight-space-programming-experiments): the prompts, every reply and thinking summary, the code and weights from every step, and the [MNIST protocol](https://github.com/lextoumbourou/weight-space-programming-experiments/tree/main/mnist-training). The repo also has tabular experiments (Titanic and synthetic data) that I'll write up separately.
