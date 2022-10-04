---
title: Natural Language Processing with NLP - Week 1
date: 2022-10-04 00:00
status: draft
category: reference/moocs
parent: attention-models-in-nlp
---

## Intro

* Natural Language Processing with Attention Models.
* Instructors: Lukasz and Younes.
* Course includes:
    * state-of-the-art for practical NLP.
    * learn to build models from scratch.
    * also learn to fine state pretrained models (the "new normal" for modern deep learning).

## Week Introduction

* Week covers the problem of [[Machine Translation]] using attention.
* We know that an LSTM works for short to medium sequences, but for longer sequences faces problems of vanishing gradients.
* By adding an attention mechanism, the decoder can get access to all relevant parts of input sentence regardless of length.
* Many types of attention, but this week focuses on simple attention.
* Also covered:
    * greedy decoding.
    * random sampling.
    * beam search
        * minimise bias risk when predicting the next word in a translated sentence.

## Seq2Seq

* Outline:
    * Intro to Neural Machine Translation
    * Show model that has typically been used for it: the Seq2Seq model.
    * Look at model's deficiencies and the improvements made by subsequent models.

* Neural machine translation uses an encoder and decoder to translate languages.
    * English to German for this week's assignment.

* [Seq2Seq model by Google in 2014](https://arxiv.org/abs/1409.3215).
    * Takes a sequence of words (or any sequence you can encode as tokens) and return another sequence.
    * Works by mapping variable length sequences to fixed length memory called [[Embedding Space]].
    * Inputs and outputs don't need to be the same length.
    * LSTMs and GRUs can deal with vanishing and exploding gradients.
    * How it works
        * Encoder takes word tokens as inputs.
        * Returns a hidden state as output.
        * Hidden state is used by decoder to generate decoded sequence.
        ![Seq to seq encoder and decoder](_media/s2s-encoder-decoder.png)
    * Encoder
        * Has an embedding layer that converts input tokens to embedding vectors.
        * At each input step, LSTM gets input embedding and hidden state from previous step.
        * Encoder returns the final hidden state, which aims to encodes the meaning of the sequence.
        ![Seq to seq encoder](_media/s2s-encoder.png)

    * Decoder
        * Constructed of embedding layer and LSTM layer.
        * Use output of last hidden step of encoder and embedding for start token token `<sos>`.
        * Model outputs most probable next work, then pass the LSTM hidden state and embedding for predicted word to the next step.
            ![Seq2Seq Decoder](_media/s2s-decoder.png)

* Seq2Seq limitations:
    * Information bottleneck:
        * Since it uses a fixed length memory for hidden states, long sequences are difficult
        * As sequence size increases, model performance decreases.
        * Could work around using all encoder hidden states? Would have memory issues with bigger sequences.
        ![Information bottleneck](_media/s2s-info-bottle-neck.png)
* Attention:
    * Maybe model can learn what's important to focus on at each step?
    * More in next section.
    ![Seq 2 Seq attention idea](_media/s2s-attention.png)
## Seq2Seq Model with Attention

* What we now call attention was introduced in paper: [Neural Machine Translation By Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* In the paper, they proved that Seq2Seq could be significantly improved with Attention by comparing BLEU scores against alternatives.
    * Note that the RNNsearch-50 model has no dropoff as the sequence length increases.
![Seq2Seq performance](_media/seq2seq-performance.png)


* Attention motivation
    * We know traditional seq2seq models use final hidden state of encoder as input to decoder.
    * Therefore, the encoder has to compress all the information into one hidden state.
    * Instead of compressing, what about concatenating the hidden states and passing to the decoder?
        * It is inefficient and likely memory prohibitive for long sequences.
     ![Use all hidden states](_media/seq2seq-concat-hidden-states.png)
    * What about combining all hidden states into one vector: "The Context Vector"?
        * Pointwise additional is a simply approach. Just add up the vectors to produce another vector.
        * Now, each step in the decoder gets information about each step, but for the first words of the sentence, the latter encoder outputs are less useful.
        * Really, it's not that different from using the last hidden state.
    ![Pointwise addition](_media/seq2seq-pointwise-addition.png)
    * Maybe we can weight the encoder vectors before the addition. If the word will be more important to the decoder, we can weight it higher.
    ![Weighted sum](/_media/seq2seq-weighted-sum.png)
    * How do we learn these weights? Attention.

* How attention weights are calculated
    * Decoders previous hidden states: $s_{i-1}$ has info about previous words in the output translation.
    * So, you can compare the decoder states with each encoder states to find most important inputs.
* Attention layer
    * Goal: return a context vector that has relevant info about encoder states.
    * The first step is to calculate the alignments, E_IJ, which is a score of how well the inputs around J match the expected output its I
    * The more it does this, the higher it should be.
    * We do this using a feedforward network, expressed as: $e_{ij} = a(s_{i-1}, h_j)$
        * Where:
            * $s_{ij}$ is the decoder's previous hidden state.
            $h_j$ is the hidden states of encoder.
        * Then we Softmax the results.
    * We can treat these Softmax values as the weights of each hidden state.
    * We can now do a weighted sum of each hidden state.
    * Diagram:
     ![Attention layer overview](_media/seq2seq-attention-layer-in-more-depth.png)

## Background on seq2seq

* Recurrent models take a sequence in order and use that to output a sequence.
* Each element in the sequence has an associated computation step $t$. For example, the third element will be computed at step $t_3$.
* These models generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$.
* Because of their sequential nature, you cannot do parallelization within training examples. That becomes an issue at longer sequence lengths, as memory constraints limit batching across examples.
* Attention mechanisms have become critical for sequence modelling in various tasks, allowing modelling of dependencies without caring about their distance in input or output sequences.

## Queries, Keys, Values and Attention

* Attention was first described in 2014, since then it has been improved.
* Conceptually queries, keys and values can be thought of as a lookup table, where the query is mapped to a key which returns a value.
* In this example, when translating between French and English, "l'heure" matches "time", so we want to get the value for time.

![Queries, Keys and Values](/_media/seq2seq-queries-keys-values.png)
* Queries, keys and vectors are represented by learned embedding vectors.
* Similarity between words is called *alignment*.
    * Similarity is used for the weighted sum.
* The alignment scores are measures of how well query and keys match.
* These alignment scores are turned into weights used for weighted sum of the value vectors.
    * The weighted sum is returned as the attention vector.
* Scale dot-product attention:
    * Pack queries into matrix $Q$.
    * Pack keys and values in matricies $K$ and $V$.
    * Perform matrix multiplication between Q and K transposed: $QK^T$. That gives you a matrix of alignments scores.
    * Then scale using the square root of the key-value dimension $d_k$: $\frac{QK^T}{\sqrt{d_k}}$
        * A regularisation step that improves performance of larger models
    * Apply Softmax to scaled scores so that the weights of each query sum to 1.
    * Finally, the weights and value matrices are multiplied to get the Attention vectors for each query.
* Total operations of scaled dot-product attention: 2 matrix multiplications and a Softmax.
* When the attention mechanism assigns a higher attention score to a word in the sequence, it means the next word in decoder's output will be strongly influenced.

![Scaled-dot product](/_media/seq2seq-scaled-dot-product.png)

* Alignment between source and target languages must be learnt elsewhere.
* Typically, alignment is used in other input embeddings before attention layer.
* The alignments weights form a matrix with source words (queries) as the rows, and targets (keys) as the columns.
    ![Alignment weights](/_media/attention-alignment-weights.png)

* Similar words will have larger weights.
* Through training, the model learns which words have similar information and encodes it into vectors.
* It's particularly useful for languages with different grammatical structures.

![Flexible Attention](/_media/seq2seq-flexible-attention.png)

* This flexibility allows the model to focus on different input words, despite the word order.
* Summary:
    * The Attention layer let's the model figure out which words are most important for the decoder.
    * Attention models use Queries, Keys and Values for creating Attention vectors.
    * Allows the model to translate on languages with different grammatical structure.

