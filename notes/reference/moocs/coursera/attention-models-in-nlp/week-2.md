---
title: Natural Language Processing with NLP - Week 2
date: 2022-10-07 00:00
status: draft
category: reference/moocs
parent: attention-models-in-nlp
---

# Week 2

## Week Introduction

* All about Transformer network.
    * Introduced in paper [Attention Is All Your Need](https://arxiv.org/abs/1706.03762).
* Despite the title, paper included convolutions.
    * These turned out not to be required.
* Many types of attention including: dot-product attention, casual attention, encoder-decoder attention and self-attention.
* This week will use the Transformer network for summarisation.

## Transformers vs RNNs

* With RNNs, sequential steps are required to encode input.
    * This means each word is dependant on the previous and it cannot be easily parallellised.
    * The longer the sequence, the slower the encoding.
* You also run more risk of vanishing gradient problems.
    * LSTMs and GRUs help a bit, but even they struggle with very long sequences.
* Attention deals with vanishing gradients by using all inputs each step.
* Transformers only rely on attention, not recurrent networks.

## Transformers Overview

* Transformer architecture was described in the [Attention Is All Your Need](https://arxiv.org/abs/1706.03762) paper in 2017
* Standard for large language models include Bert, GPT3, T5 etc.
* Uses Scaled Dot-Product Attention:

![Scaled Dot-Product](/_media/attention-scaled-dot-product.png)

* Very efficient as it's just matrix multiplication operations and a Softmax.
* Model can grow larger and more complex while using less memory.
* Transformer model uses "multi-head attention" layer:
    * has multiple linear transformations and dot-product attention running in parallel.
    * linear transations have parameters that are learned.

    ![Multi-head attention](/_media/attention-multi-head.png)

* Encoder step-by-step:
    * Multi-head attention modules perform self-attention on input sequence.
        * Every item in input attends to every other item.
        * Then residual connection and normalisation.
        * Then feed forward layer
        * Then residual connection and normalization.
    * This layer is one block that is repeated N times.
    * Gives you contextual representation of inputs.

    ![Encoder diagram](/_media/attention-encoder.png)
* Decoder:
    ![Decoder](/_media/attention-decoder.png)
    * First attention module is masked so each position only attends to previous positions: can't see into the future.
    * 2nd attention module, takes encoder outputs and decoder attends to all items.
    * Whole later repeated one after the other.
* Positional encoding:
    * Transformers don't use recurrent neural networks, so needs a way to represent word order.
    * Positional encoding vectors are added to word embeddings for input embeddings.
    * Positional encoding can be learned or fixed.
    ![Positional Encoding](/_media/attention-pos-encoding.png)

## Transformer Applications

* Transformers applications diverse: use in NLP and beyond.
* Applications:
    * text summarization.
    * named entity recognition
    * automatic question answering
    * machine translation.
    * chat-bots
    * sentiment analysis
    * market intelligence.
* State of the art models
    * GPT-2
        * Stands for generative pre-training for Transformer
    * BERT
        * Bidirectional Encoder Representations from Transformers
    * T5
        * Text-to-text transfer transformer

### T5: Text-To-Text Transfer Transformer

* Trained to do multiple tasks.
* Pass in prompts to tell the model which task to do:
    * Translation:
        * ``Translate English into French:** "I am happy"``
    * Return acceptable or not:
        * `Cola sentence: "He bought fruits and.``
        * Returns Acceptable or Unacceptable.
    * Return sentence similarity:
        * ``Stsb sentence 1: "Cats and dogs are mammals" | Sentence2: "Hello world"``
    * Q & A:
        *``Question: Which volcano in Tanzania is the highest mountain in Africa? | Answer: Mount Kilimanjaro
* T5 can also perform regression and summarisation.
* Regression example:
    * Output similarity between 2 sentences.
* Summarisation example: long sentence to shorter sentence
* T5 quiz: https://t5-trivia.glitch.me/

## Scaled and Dot-Product Attention

* The main operation in a transformer is [Scaled-Dot Product Attention](Scaled-Dot Product Attention).
* Recall that it has queries, keys and values.
* The attention layer outputs context vectors for each query.
    * These are weighted sums of the value $V$.
* Similarity between queries and keys determines weights assigned to each value.
* Softmax ensures weights add to 1.
* Division by square root of dimension of key factors improves performance.

![Softmax scaling](/_media/attention-softmax-scaling.png)

$$
\text{softmax}( \frac{QK^{T}}{\sqrt{d_k}}) V
$$

* This mechanism is efficient: relies on only matrix multiplications and [Softmax Activation Function](../../../../permanent/Softmax Activation Function.md).
* Usually run on GPUs or TPUs to speed up the training.

### Queries, Keys and Values

* Example translating French sentence: "Je suis heureux".
    * Firstly get embedding vector for each word.
    * Stack into matrix Q.
    ![Queries part of embedding](/_media/attention-queries.png)

    * Then, get the key matrix $K$, from the input sequence: "I am happy".
    * You will usually use the same key matrix for the values $V$, though they can sometimes be transformed first.
    * The number of vectors for key and values will usually be the same.
    ![Key and values](/_media/attention-key-and-values.png)

* Now compute the product using [Matrix Multiplication](../../../../permanent/matrix-multiplication.md) between $Q$ and the transposed $K$
* Then scale by inverse of square of dimension of key vectors: $\sqrt{d_k}$
    * The computation, will give you a matrix with weights for each key per query.
    * Weight matrix will have total number of elements equal to: `(num_queries X num_keys)`.
![Attention math](/_media/attention-math.png)
    * The weight matrix is multiplied by the value matrix, $V$, to return a context vector for each query.

## Masked Self-Attention

* Outline of lesson:
    * Review different types of attention mechanisms used in Transformer model.
    * Learn how to compute Masked Self-Attention.

### Encoder-Decoder Attention

* Style of attention covered in course so far:
    * Queries from one sentence, keys and values from another.
    ![Encoder-Decoder Attention](/_media/attention-encoder-decoder-attention.png)

* Used in translation task from last week.

### Self-Attention

* In Self-Attention, queries, keys and values come from the same sentence.
* This gives you contextual representations of your words.
![Self Attention](/_media/attention-self-attention.png)

### Masked Self-Attention

* In Masked Self-Attention, queries, keys and values come from the same sentence. However, queries don't attend to future positions.
* Same as masked self-attention, howevera mask matrix is added within the Softmax.
    * Has 0s on all positions except for elements above the diagonal, which are $-\inf$
    * After taking a Softmax, all elements in weights matrix are zero for keys in subsequent position to query.
* Multiple weights by value matrix to get context vectors per query as normal.
* We put this mechanism in decoder. It ensures that all predictions only depend on known outputs.
![Masked Self-Attention](/_media/attention-masked-self-attention.png)

## Multi-head Attention

* One improve that can be made on the attention mechanisms discussed so far is [Multi-Head Attention](Multi-Head Attention).
    * Apply multiple versions of Scaled-Dot Product Attention in parallel.
    * Then apply a transformations many times.
    * Different sets of representation, allows model to return multiple relationships betweens words in query and key matrice.
    ![Plot embeddings in Scaled-Dot Product Attention](/_media/attention-plot-embeddings.png)

    ![Multi-head attention](/_media/attention-multi-head.png)

* How it works:
    * Inputs are multiple sets of values, keys and queries matrices.
    * Applied Scaled Dot-Product Attention to each set.
    * Concat results from each head into a single matrix.
    * Transform concat results into single result using Linear layer.
* In the original paper, the author sets the size of the query, key and value matrices to be `d_model / num_heads`.
    * This means that the computation costs aren't much more significant than a single head.

## Multi-head Attention (Reading)

* Intution behind multi-head attention:
    * Each head has a different linear transformer to represent words.
    * That means, different heads can learn different relationships between words.
* The context vector is concatenated for each head to create a final vector for each word.

## Transformer Decoder

* Decoder gets input sequence of a tokenised sentence as input.
* For each token, an embedding is fetched.
* Add these word embeddings to positional embeddings.
* Pass these word embeddings to multi-head attention.
* Then feed to Attention Layer which processes each position independently.
* Pass to residual connection with layer normalization.
* Repeat Attention and Feed Forward N times depending on model size.

![Transformer Decoder](/_media/attention-transformer-decoder.png)

* More detail:
    * Model has 3 layers at the beginning: inputs, word embedding and positional embedding.
        * The word embedding and positional embeddings are learned.
    * Shift right introduced start token `<start>`, so model can predict first word.
    * After word embedding, shape goes from `(batch, length)` to `(batch, length, d_model)`.
        * `d_model` refers to embedding size which is usually 512, 1024 and sometimes 10k+.
    * After word and position embeddings, you pass to N Decoder Blocks.
    * Next a fully connected layer that outputs tensors: `(batch, length, vocab size)`.
    * And a Log Softmax for cross-entropy loss.
    * How the Decoder Blocks are built:
        * Set of word + position embeddings are passed in multi-head attention model.
        * Attention searches position in sequence to identify relationships.
        * Each words on sequence is weighted.
        * Then residual connection.
        * Followed by layer normalisation step to speed up training.
        * Each word is passed through feed forward layer: ie embeddings are fed into neural network.
        * Dropout at the end for regularisation.
        * Layer norm step is repeated capital N times.
        * Finally, encoder layer output is obtained.
        * After the attention mechanism and the normalization step, some non-linear transformations are introduced by including fully connected feed forward layers, with Relu activations for each inputs.
            * Uses shared parameters for efficiency.
        * The feed forward output vectors are like the hidden states of the original RN encoder.

## Transformer Summarizer

* This week's assignment:
    * Input: entire news articles.
    * Output: Summary of the article: a few sentences that mention the most important ideas.
* Setup for summarisation problem:
    * Concat the input, ie the article, with summary.
    * Separate with `<EOS>` tokens.
    * Example:

    ```
    ARTICLE TEXT <EOS> ARTICLE SUMAMRY <EOS> <pad>
    ```

    * Input is tokenised as a sequence of integers:

    ```
    [2, 3, 5, 2, 1, 3 ... 0]
    ```

    * In this example 0 denotes padding, and 1 EOS token.
    * We are not concerned with predicting input sequence, so use weighted loss:
        * Loss is weighted 0 until the first EOS tag, then start 1 for the summary tokens.
        * Weighted loss: $J = -\frac{1}{m} \sum\limits^{m}_i \sum\limits^{K}_{j} y^{i}_{j} \log \hat{y}^{i}_{j}$
        * In other words, cross-entropy loss over just the summary sentence.
        * Note: if there's low information in the summary, you can weigh the loss tokens with something other than 1s, like 0.1, 0.2, 0.5 etc.
* Inference with a language model:
    * At inference time, you will input the article with the `EOS` word and keep asking for words until you get the final `EOS` token.
    * You get a probability distribution for each word, so you can pick the word by random sampling.
        * You will get a different summary each time you run inference.
