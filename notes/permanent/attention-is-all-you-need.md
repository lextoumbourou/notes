---
title: Attention Is All You Need
date: 2023-12-04 00:00
modified: 2023-12-04 00:00
status: draft
---

Shell page for [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.

![](../../../_media/attention-is-all-you-need-title.png)
![](../../../_media/attention-is-all-you-need-abstract.png)

## Abstract

At the time of the paper, the prevalent sequence-to-sequence were [Encoder-Decoder](encoder-decoder.md) models that used recurrent or convolutional neural networks, where an encoder would represent an input sequence of tokens as a sequence of embeddings, and a decoder would take those embeddings and predict a new sequence one token at a time. The best performance models would connect the encoder and decoder using an [Attention Mechanism](attention-mechanism.md) mechanism.

This paper proposes a revolutionary architecture called [Transformer](transformer.md), entirely based on attention mechanisms, without recurrence and convolutions at all.

The model achieved:

* 28.4 [Bilingual Evaluation Understudy](../../../permanent/bilingual-evaluation-understudy.md) on the [WMT 2014 English-to-German](WMT%202014%20English-to-German) translation task, improving over the existing best results, including ensembles. by over 2 [BLEU](BLEU).
 * On [WMT 2014 English-to-French](WMT%202014%20English-to-French) translation task, the model establishes a new single-model state-of-the-art BLEU score of 41.8 after 3.8 days of training on eight GPUs. That was a fraction of the training codes from literature.

They also show the Transformer generalised well to other tasks.

## Introduction

[Recurrent Neural Networks](../../../permanent/recurrent-neural-networks.md), [LSTM](lstm.md) and [Gated Recurrent Neural Networks](Gated%20Recurrent%20Neural%20Networks) were the go-to models for state-of-the-art performance in sequence modelling tasks, like building a [Language Model](language-model.md) and models for [Machine Translation](Machine%20Translation).

All these model "factor computation along the symbol positions of the input and output sequences". They aliging the positions to steps in computation time, they generate a sequence of hidden states, $h_t$, which are calculated as functions of the previous hidden states $h_{t-1}$ and input position $t$.

Because of this sequential natural, we cannot parallelization within training examples, which precludes working on longer sequences, as we often won't have enough memory to batch across examples. Various improvements have been proposed like factorization tricks and conditional computation, which do improve the performance, but still the problem of sequential computation remains.

[Attention Mechanism](attention-mechanism.md) mechanisms are a key part of sequence modelling, allowing modelling of dependencies without regard to their distance in input or output sequences. Except for a few examples, attention was used with a recurrent network.

This paper proposes the Transformer, a model architecture that doesn't have recurrence and relies purely on attention for global dependencies for input and output.

Transformer allows for more parallelisation and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

## Background

The goal of reducing sequential computation was explored in [Convolutional Neural Network](convolutional-neural-network.md) architectures like [Extended Neural GPU](../../../permanent/extended-neural-gpu.md), [ByteNet](ByteNet) and [ConvS2S](ConvS2S), which aimed to use CNNs to compute the hidden representations in parallel for all input and output positions.

In these models, the number of operations required to relate signals from two arbitrary input or output positions grows
in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet.

This makes it more difficult to learn dependencies between distant positions.

In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

[Self-Attention](self-attention.md), sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization,
textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.

In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

## Model Architecture

Most competitive neural sequence transduction models have an [Encoder-Decoder](encoder-decoder.md) structure, where we map an input sequence of symbol represenations $(x_1, \ldots, x_n)$ to a sequence of continuous representations $z = z_1, ..., z_n$.

Then we feed $\mathbf{z}$ into a decoder, which generates an output sequence $y_1, ..., y_m$ one element at a time.

At each step the model is [Auto-Regressive](Auto-Regressive), in that it takes the previously generates symbols as input when generating the next token.

### Encoder and Decoder Stacks

**Encoder**

The encoder is composed of a stack of N = 6 identical layers.

Each layer has two sub-layers:
* One [Multi-head Attention](multi-head-attention.md) mechanism.
* Position-wise fully connected feed-forward network.

They employ a residual connection around each of the two sub-layers, followed by layer normalisation.

The output of the sublayer uses [Layer Norm](../../../permanent/layer-norm.md) is `LayerNorm(x + Sublayer(x))` where Sublayer is the function implemented by the sub-layer itself.

All sublayers produce outputs of dimension $d_{\text{model}} = 512$

**Decoder**

The decoder also has a stack of $N = 6$ identical layers. In the decoder has a third -sub-layer that does multi-head attention over the output of the encoder stack.

They have residual connections around each sub-layer followed by layer normalisation.

They modify the self-attention sub-layer in the decoder stack to prevent positions from attention to subsequent positions.

This masking, combined with the fact the output embeddings are offset by one position, ensures that the predictions for position $i$ can only depend on the known outputs at positions less than i.

### Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

#### Scaled Dot-Product Attention

![](../../../_media/attention-is-all-you-need-scaled-dot-product-attention.png)
*Partial Figure 2 from paper Attention Is All Your Need*

We call our particular attention [Scaled-Dot Product Attention](scaled-dot-product-attention.md).

The input consists of queries and keys of dimension $dk$, and values of dimension $dv$.

Step by step:

1. We compute the dot products of the query with all keys.
2. divide each by $\sqrt{dk}$.
3. apply a softmax function to obtain the weights.
4. Perform a dot product on the weights and values to generate the final representation.

In practice, they compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$.

The keys and values are also packed together into matrices $K$ and $V$ . We compute the matrix of outputs as:

$\text{Attention}(Q, K, V ) = \frac{\text{softmax}(Q @ K^{T})}{\sqrt{d_k}}$

The two most commonly used attention functions are [Additive Attention](Additive%20Attention), and [Dot-Product Attention](Dot-Product%20Attention).

Dot-product attention is identical to our algorithm, except for the scaling factor of $\sqrt{d_k}$.

Additive attention computes the compatibility function using a feed-forward network with
a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
much faster and more space-efficient in practice, since it can be implemented using highly optimized
matrix multiplication code.

While for small values of $dk$ the two mechanisms perform similarly, additive attention outperforms
dot product attention without scaling for larger values of $dk$.

They think that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. Hence adding the scaling term: $\frac{1}{\sqrt{d_k}}$

#### Multi-head Attention

Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
we found it beneficial to linearly project the queries, keys and values h times with different, learned
linear projections to dk, dk and dv dimensions, respectively.

On each of these projected versions of
queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional
4To illustrate why the dot products get large, assume that the components of q and k are independent random
variables with mean 0 and variance 1. Then their dot product, q · k =
Pdk
i=1 qiki, has mean 0 and variance dk.
4
output values. These are concatenated and once again projected, resulting in the final values, as
depicted in Figure 2

Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this.

MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
where headi = Attention(QWQ
i
, KW K
i
, V WV
i
)

Where the projections are parameter matrices W
Q
i ∈ R
dmodel×dk , W K
i ∈ R
dmodel×dk , WV
i ∈ R
dmodel×dv
and WO ∈ R
hdv×dmodel
.

In this work we employ h = 8 parallel attention layers, or heads. For each of these we use
dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost
is similar to that of single-head attention with full dimensionality

#### Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

* In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
* The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
* Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.
    * We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.
    * We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.

### Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
connected feed-forward network, which is applied to each position separately and identically. This
consists of two linear transformations with a ReLU activation in between.

FFN(x) = max(0, xW1 + b1)W2 + b2

While the linear transformations are the same across different positions, they use different parameters
from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
df f = 2048.

### Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input
tokens and output tokens to vectors of dimension dmodel.

We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities

In
our model, we share the same weight matrix between the two embedding layers and the pre-softmax
linear transformation, similar to [30].

In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$

In table 1, they show maximim path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the equence lrngth, $d$ is the represetation dimensino, $k$ is the key size of convolutions and $r$ the size of the neighbourhood in restircted self-attention.

| Layer Type                  | Complexity Per Layer     | Sequential Operations | Maximum Path Length |
| --------------------------- | ------------------------ | --------------------- | ------------------- |
| Self-Attention              | $O(n^2 \cdot d)$         | $O(1)$                | $O(1)$              |
| Recurrent                   | $O(n \cdot d^2)$         | $O(n)$                | $O(n)$              |
| Convolutional               | $O(k \cdot n \cdot d^2)$ | $O(1)$                | $O(\log_k(n))$      |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$   | $O(1)$                | O(n/r)<br>          |

### Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the
order of the sequence, we must inject some information about the relative or absolute position of the
tokens in the sequence

To this end, we add "positional encodings" to the input embeddings at the
bottoms of the encoder and decoder stacks

 The positional encodings have the same dimension dmodel
as the embeddings, so that the two can be summed. There are many choices of positional encodings,
learned and fixed [9].

In this work, we use sine and cosine functions of different frequencies:

$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_model}$
$PE(pos, 2_i + 1) = \cos(pos/10000^{2i/d_model})$

where pos is the position and i is the dimension.

That is, each dimension of the positional encoding corresponds to a sinusoid.

The wavelengths form a geometric progression from 2π to 10000 · 2π

We
chose this function because we hypothesized it would allow the model to easily learn to attend by
relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of
P Epos.

We also experimented with using learned positional embeddings [9] instead, and found that the two
versions produced nearly identical results

We chose the sinusoidal version
because it may allow the model to extrapolate to sequence lengths longer than the ones encountered
during training

## Why Self-Attention?

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations
(x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi
, zi ∈ R
d
, such as a hidden
layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we
consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can
be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network

Learning long-range dependencies is a key challenge in many sequence transduction tasks.

One key factor affecting the
ability to learn such dependencies is the length of the paths forward and backward signals have to
traverse in the network. The shorter these paths between any combination of positions in the input
and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare
the maximum path length between any two input and output positions in networks composed of the
different layer types

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially
executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of
computational complexity, self-attention layers are faster than recurrent layers when the sequence
6
length n is smaller than the representation dimensionality d, which is most often the case with
sentence representations used by state-of-the-art models in machine translations, such as word-piece
[38] and byte-pair [31] representations

To improve computational performance for tasks involving
very long sequences, self-attention could be restricted to considering only a neighborhood of size r in
the input sequence centered around the respective output position. This would increase the maximum
path length to O(n/r). We plan to investigate this approach further in future work.

A single convolutional layer with kernel width k < n does not connect all pairs of input and output
positions. Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels,
or O(logk(n)) in the case of dilated convolutions [18], increasing the length of the longest paths
between any two positions in the network. Convolutional layers are generally more expensive than
recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity
considerably, to O(k · n · d + n · d
2
). Even with k = n, however, the complexity of a separable
convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer,
the approach we take in our model.

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions
from our models and present and discuss examples in the appendix. Not only do individual attention
heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic
and semantic structure of the sentences.

## Training

### Training Data and Batching

They trained on the standard WMT 2014 English-German dataset consisting of 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulurty of 37k tokens.

For English-French, we used the significantly larger WMT
2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece
vocabulary [38]

 Sentence pairs were batched together by approximate sequence length. Each training
batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000
target tokens

### Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using
the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We
trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the
bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps
(3.5 days).

### Optimizer

We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9
. We varied the learning
rate over the course of training, according to the formula:
lrate = d
−0.5
model · min(step_num−0.5
, step_num · warmup_steps−1.5
) (3)
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.

### Regularisation

We employ three types of regularization during training:

Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the
sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
Pdrop = 0.1.

Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 [36]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

## Results

### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)
in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0
BLEU, establishing a new state-of-the-art BLEU score of 28.4.

The configuration of this model is
listed in the bottom line of Table 3.

Training took 3.5 days on 8 P100 GPUs. Even our base model
surpasses all previously published models and ensembles, at a fraction of the training cost of any of
the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,
outperforming all of the previously published single models, at less than 1/4 the training cost of the
previous state-of-the-art model. The Transformer (big) model trained for English-to-French used
dropout rate Pdrop = 0.1, instead of 0.3.

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which
were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We
used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters
were chosen after experimentation on the development set. We set the maximum output length during
inference to input length + 50, but terminate early when possible [38].

Table 2 summarizes our results and compares our translation quality and training costs to other model
architectures from the literature. We estimate the number of floating point operations used to train a
model by multiplying the training time, the number of GPUs used, and an estimate of the sustained
single-precision floating-point capacity of each GPU 5
.
