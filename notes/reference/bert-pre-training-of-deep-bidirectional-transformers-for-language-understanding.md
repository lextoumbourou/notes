---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
date: 2024-10-13 00:00
modified: 2026-09-26 08:55
status: draft
---

Paper that introduced language representation model [BERT](../permanent/bert.md), which stands for Bidirectional Encoder Representations from Transformers.

The major contribution of this paper is building upon language representation models [ELMo](../permanent/elmo.md) and [GPT](../permanent/gpt.md), but BERT uses *bidirectional* representations by jointly conditioning on *both left and right context* in all layers. Where prior work used a next token prediction, which requires a left-to-right architecture, by using a [Masked Language Modelling](../permanent/masked-language-modelling.md) pre-training objective they can make the model bidirectional. This technique was inspired by [Cloze procedure: A new tool for measuring readability](../permanent/cloze-procedure-a-new-tool-for-measuring-readability.md).

Pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

It obtains new state-of-the-art results on eleven natural language processing tasks including:

* pushing the [GLUE](../permanent/glue.md) score to 80.5% (7.7% point absolute improvement)
* MultiNLI accuracy to 86.7% (4.6% absolute improvement)
* SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement)
* SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

## Architecture

BERT’s model architecture is a multi-layer bidirectional [Transformer](../permanent/transformer.md) encoder based on the original implementation in the tensor2tensor library.

They denote the number of layers (i.e., Transformer blocks) as $L$, the hidden size as $H$, and the number of self-attention heads as $A$.

We primarily report results on two model sizes:

- $\text{BERTBASE}(L=12, H=768, A=12, \text{Total Parameters}=110M)$
- $\text{BERTLARGE}(L=24, H=1024, A=16, \text{Total Parameters}=340M)$

$\text{BERTBASE}$ was chosen to have the same model size as OpenAI GPT for comparison purposes.

However, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.

**Input/Output Representations**

To make BERT
handle a variety of down-stream tasks, our input
representation is able to unambiguously represent
both a single sentence and a pair of sentences
(e.g., ⟨Question, Answer⟩) in one token sequence.
Throughout this work, a “sentence” can be an arbitrary span of contiguous text, rather than an actual
linguistic sentence. A “sequence” refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.
We use WordPiece embeddings (Wu et al.,
2016) with a 30,000 token vocabulary. The first
token of every sequence is always a special classification token (`[CLS]`). The final hidden state
corresponding to this token is used as the aggregate sequence representation for classification
tasks. Sentence pairs are packed together into a
single sequence. We differentiate the sentences in
two ways. First, we separate them with a special
token (`[SEP]`). Second, we add a learned embedding to every token indicating whether it belongs
to sentence A or sentence B. As shown in Figure 1,
we denote input embedding as $E$, the final hidden
vector of the special `[CLS]` token as $C \in \mathbb{R}^H$,
and the final hidden vector for the $i^{th}$ input token
as $T_i \in \mathbb{R}^H$.
For a given token, its input representation is
constructed by summing the corresponding token,
segment, and position embeddings. A visualization of this construction can be seen in Figure 2.

## Pre-training

They use the BooksCorpus (800M words) and English Wikipedia (2,500M words). For Wikipedia they extract only the text passages and ignore lists, tables, and headers.

They mention it's important to use document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark to extract long contiguous sequences.

**Task #1: Masked LM**

Intuitively, it is reasonable to believe that a deep bidirectional model is
strictly more powerful than either a left-to-right
model or the shallow concatenation of a left-to-right and a right-to-left model. Unfortunately,
standard conditional language models can only be
trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially
predict the target word in a multi-layered context.
In order to train a deep bidirectional representation, we simply mask some percentage of the input
tokens at random, and then predict those masked
tokens. We refer to this procedure as a “masked
LM” (MLM), although it is often referred to as a
Cloze task in the literature (Taylor, 1953). In this
case, the final hidden vectors corresponding to the
mask tokens are fed into an output softmax over
the vocabulary, as in a standard LM. In all of our
experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to
denoising auto-encoders (Vincent et al., 2008), we
only predict the masked words rather than reconstructing the entire input.
Although this allows us to obtain a bidirectional pre-trained model, a downside is that we
are creating a mismatch between pre-training and
fine-tuning, since the `[MASK]` token does not appear during fine-tuning. To mitigate this, we do
not always replace “masked” words with the actual `[MASK]` token. The training data generator
chooses 15% of the token positions at random for
prediction. If the i-th token is chosen, we replace
the i-th token with (1) the `[MASK]` token 80% of
the time (2) a random token 10% of the time (3)
the unchanged i-th token 10% of the time. Then,
$T_i$ will be used to predict the original token with
cross entropy loss. We compare variations of this
procedure in Appendix C.2.

**Task #2: Next Sentence Prediction (NSP)**

Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order
to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically,
when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual
next sentence that follows A (labeled as IsNext),
and 50% of the time it is a random sentence from
the corpus (labeled as NotNext). As we show
in Figure 1, $C$ is used for next sentence prediction (NSP). Despite its simplicity, we demonstrate in Section 5.1 that pre-training towards this
task is very beneficial to both QA and NLI. The NSP task is closely related to representation-learning objectives used in Jernite et al. (2017) and
Logeswaran and Lee (2018). However, in prior
work, only sentence embeddings are transferred to
down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters.

## Fine-tuning BERT

Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks
(whether they involve single text or text pairs) by
swapping out the appropriate inputs and outputs.
For applications involving text pairs, a common
pattern is to independently encode text pairs before applying bidirectional cross attention, such
as Parikh et al. (2016); Seo et al. (2017). BERT
instead uses the self-attention mechanism to unify
these two stages, as encoding a concatenated text
pair with self-attention effectively includes bidirectional cross attention between two sentences.
For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end. At the input, sentence A and sentence B from pre-training
are analogous to (1) sentence pairs in paraphrasing, (2) hypothesis-premise pairs in entailment, (3)
question-passage pairs in question answering, and
(4) a degenerate text-∅ pair in text classification
or sequence tagging. At the output, the token representations are fed into an output layer for token-level tasks, such as sequence tagging or question
answering, and the `[CLS]` representation is fed
into an output layer for classification, such as entailment or sentiment analysis.

Compared to pre-training, fine-tuning is relatively inexpensive. All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting
from the exact same pre-trained model. We describe the task-specific details in the corresponding subsections of Section 4. More details can be
found in Appendix A.5.
