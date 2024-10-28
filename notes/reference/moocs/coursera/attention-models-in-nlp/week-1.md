---
title: Natural Language Processing with NLP - Week 1
date: 2022-10-04 00:00
status: draft
category: reference/moocs
parent: attention-models-in-nlp
---

Notes from [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing) offered by deeplearning.ai.

## Intro

* Natural Language Processing with Attention Models.
* Instructors: Lukasz and Younes.
* Course includes:
    * state-of-the-art for practical NLP.
    * learn to build models from scratch.
    * also learn to fine state pretrained models (the "new normal" for modern deep learning).

## Week Introduction

* Week covers the problem of [Machine Translation](Machine Translation) using attention.
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
    * Works by mapping variable length sequences to fixed length memory called [Embedding Space](Embedding Space).
    * Inputs and outputs don't need to be the same length.
    * [lstm](../../../../permanent/lstm.md) and [Gated Recurrent Unit](../../../../permanent/gated-recurrent-unit.md) architectures can deal with vanishing and exploding gradients.
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

![Queries, Keys and Values](_media/seq2seq-queries-keys-values.png)
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

![Scaled-dot product](_media/seq2seq-scaled-dot-product.png)

* Alignment between source and target languages must be learnt elsewhere.
* Typically, alignment is used in other input embeddings before attention layer.
* The alignments weights form a matrix with source words (queries) as the rows, and targets (keys) as the columns.

    ![Alignment weights](_media/attention-alignment-weights.png)

* Similar words will have larger weights.
* Through training, the model learns which words have similar information and encodes it into vectors.
* It's particularly useful for languages with different grammatical structures.

![Flexible Attention](_media/seq2seq-flexible-attention.png)

* This flexibility allows the model to focus on different input words, despite the word order.
* Summary:
    * The Attention layer let's the model figure out which words are most important for the decoder.
    * Attention models use Queries, Keys and Values for creating Attention vectors.
    * Allows the model to translate on languages with different grammatical structure.

## Setup for Machine Translation

* Example of input data for assignment:

| English                    | French                              |
| -------------------------- | ----------------------------------- |
| I am hungy!                | J'ai faim!                          |
| ...                        | ...                                 |
| I watched the soccer game. | J'ai regardé le match de football. |

* Note: the dataset used isn't entirely clean.
* Machine translation setup:
    * Usually would use pre-trained vector embeddings.
    * However, for assignment will use one-hot vectors to represent words.
    * Keep track of index mapping with `word2ind` and `ind2word` dictionaries.
    * Add end of sequence tokens: `<EOS>` then pad token vectors with zeros to ensure batches are the same length.

## Teacher Forcing

* seq2seq models work by feeding output of decoder of previous step as input.
    * This means, there's no set length for the output sequence.
* When training the model, you compare the decoder output sequence with the output sequence.
    * Calculating [Cross-Entropy](../../../../permanent/cross-entropy.md) loss for each step, then summing the steps together for the total loss.
* In practice, this is an issue in the early stages of training, as model will make many wrong predictions.
* The problem compounds as model keeps making wrong predictions, making target sequence very far from translated sequence.
* To avoid this problem, you can use ground-truth words as inputs to the decoder.
    * This means, even if the model makes a wrong prediction, it pretends as if the model has made a correct one.
* It's common to slowly start using decoder outputs over time, so that you are eventually no longer feeding in the target words.
    * This is called: [Curriculum Learning](Curriculum Learning).

## NMT Model with Attention

* Training from scratch
    * Pass input sequence through encoder.
    * The decoder passes its hidden state to the Attention Mechanism. Since it's difficult to implement, we use 2 decoders, one pre attention and one after attention.

    ![NMT Model](_media/nmt-model.png)
* First step: create 2 copies of the input tokens and target topics.
    * First copy of input goes into Encoder.
    * First copy of target, goes into Pre-attention Decoder. These are shifted right and we add a start of sentence token.
* Input encoder embedding is fed into LSTM, outputting Queries and Keys
* Pre-attention is also fed into LSTM, outputting Values.
* We use these to generate Attention Context Vectors.
* Then Decoder takes copy of target tokens, and is fed them with context vectors, to generate log probabilities.

## BLEU Score

* Evaluates quality of machine translated text by comparing a candidate translation to one or more references, which are often human translations.
* 1 is best, 0 is worst.
* Calculated by computing the precision of candidates by comparing it n-grams with a reference translation.
* Example:

| Sample Type | Sample                   |
| ----------- | ------------------------ |
| Candidate   | I, I am I                |
| Reference 1 | Younces said I am hungry |
| Reference 2 | He said I am hungry      |

* If using unigrams, you would count how many words from candidate appears in any of the referneces, and divide by count of total number of words in the candidate translation.
* You can view as a precision metric.
* That means that in this example, each word in the candidate would have a value of 1, and the total number of words in candidate is 4.
* So it would have a perfect BLEU score, which obviously isn't right.
* A modified version is that you remove the word from the reference after matching in the candidate.
* So you match `I`, then remove it from both references. Then the next time you match `I`, it won't be in the reference, so it's not a match and so on. That would give you a Bleu score of 0.5.
* The most widely used loss function for machine translations.
* Main drawback is that it doesn't consider semantic meaning.

## ROUGE-N Score

* A performance metric for assessing the performance of Machine Translation models.
* Comes from a family of metrics called ROUGE, which stands for: *Recall-Oriented Understudy for Gisting Evaluation*.
* From the name we can infer that it's a recall-oriented metric (as opposed to the precision oriented BLEU). Which means, it's concerned with how many of the reference translations appear in the candidate.
* Compares candidates n-grams with reference (human) translations.

| Sample Type | Sample                   |
| ----------- | ------------------------ |
| Candidate   | I, I am I                |
| Reference 1 | Younces said I am hungry |
| Reference 2 | He said I am hungry      |

* Since there is 2 references, you would end up with 2 ROUGE-N scores.
* Unigram example:

    Count 1: Younces=0, said=0, I=1, am=1, hungry=0,
        Total: 2/5 = 0.4
    Count 2: He=0, said=0, I=1, am=1, hungry=0
        Total: 2/5 = 0.4

* You can calculate the F1 score if you want to combine both metrics:

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precicision} + \text{Recall}} \Rightarrow 
F1 = 2 \times \frac{\text{BLEU} \times \text{ROUGE-N}}{\text{BLEU} + \text{ROUGE-N}}
$$

* In the examples we've used, the result would be:

$$
F1 = 2 \times \frac{0.5 \times 0.4}{0.5 + 0.4} = \frac{4}{9} \approx 0.44
$$

* All evaluations metrics so far don't consider sentence structure and semantics.

## Sampling and Decoding

2 ways to construct a sentence from model outputs:
1. Greedy decoding.
2. Random sampling.

Seq2seq refresher:
1. Output of a seq2seq model is the output of the model's Dense layer fed into Softmax or Log Softmax.
2. You now have a probability distribution over all words and symbols in target vocabulary.
3. The final output of the model depends on how you choose the words from the distribution.

Simplest approach is [Greedy Decoding](Greedy Decoding): select the most probable word at each step.

The downsides to this approach is that the greedy decoder can give you repeated tokens for the most common words. Though it works for shorter sequences in practice.

Another approach is [Random sampling](Random sampling) from the distribution. However, it can return results that are too random. You can mitigate this a bit by assigining higher weights to more probable words.

[Temperature](Temperature) that can be tuned if you want more or less randomness in predictions.

A lower temperature setting will give you a more confident yet conservative set of outputs. Higher temperature gives you a more "excited", random network.

Here's the setting from the GPT3 playground:

![GPT Temperature](_media/gpt-temperature.png)

Both of these methods don't always produce the most convincing outputs, compared to those coming up in future lessons.

## Beam Search

* Beam Search finds the best sequences over a fixed window size known as [Beam Width](Beam Width).
* The methods covered earlier only consider single word probabilities at a time. However, the most probable translation given an input usually isn't the one that selects the most probable word at each step. Especially at the start of a sequence, choosing the most probable word can lead to an overall worse translation.
* Given infinite compute power, you could calculate probabilities of each possible output sequence and choose the best. In practice, you can use [Beam Search](Beam Search).
* Beam search finds the most likely output sentence by chooing a number of best sequences based on conditional probabilities at each step.
* Step by step:
    1. At each step, calculate the probability of multiple possible sequences.
    2. Beam width B determines number of sequences you keep.
        1. Continue until all B most probable sequences end with `<EOS>`.
    3. Greedy decoding is just Beam search with B=1.
* Problems with Beam Search:
1. It penalises long sequences, so you have to normalise by length.
2. Computationally expensive and uses lots of memory.

## Minimum Bayes Risk

Step-by-step:

1. Generate candidate translations.
2. Assign a similarity to every pair using a similarity score (like ROUGE)
3. Select sample with the highest average similarity.

If using Rouge, MBR would be summarised like this:

![MBR with Rouge](_media/attention-mbr-rouge.png)

This method gives better performance than random sampling and greedy decoding.

## Stack Semantics in Trax

### `t1.Serial` combinator is stack oriented

* Recall that a stack is a data structure that follows Last In, First Out (LIFO) principle.
    * Whatever element is pushed onto the stack will be the first one popped out.

Creating an addition and multiplication layer:

```python
from trax import layers as tl

def Addition():
    layer_name = 'Addition'

    def func(x, y):
        return x + y

    return tl.Fn(layer_name, func)

def Multiplication():
    layer_name = 'Multiplication'

    def func(x, y):
        return x * y

    return fl.Fn(layer_name, func)
```

Implement the computations using Serial combinator:

```python
serial = tl.Serial(
    Addition(), Multiplication(), Addition()
)

# Add 3 + 4, multiply result by 15 and add 3
x = (np.array([3]), np.array([4]), np.array([15]), np.array([3]))
serial.init(shapes.signature(x))
```

### `tl.Select` combinator in context of serial combinator

If we wanted to make a calculate ```(3 + 4) * 3  + 4```, we can use `Select` to perform the calculation like:

1. 4
2. 3
3. tl.Select([0, 1, 0, 1])
4. add
5. mul
6. add

The `tl.Select` requires a list or tuple of 0-based indices to select elements relative to the top of the stack. Since the top of the stack is 3 (at index 0) and 4 (at inde 1), after the select operation the stack has: 3, 4, 3, 4.

Then, the add operation pops the first 2 selects off the stack and replaces with 3+4=7, then multiplies that by 3 to give 21. Finally adds 4 to that to give us 25.

The `n_in` argument to `tl.Select` tells us how many things to pop off the stack and replace with the index.

```
tl.Select([0], n_in=2)
```

takes 2 elements off the stack and replaces them with just the first.

```tl.Residual``` creates a skip connection around `Addition` in this example:

```
serial = tl.Serial(
    tl.Select([0, 1, 0, 1]),
    tl.Residual(Addition())
)
```

## Papers to read

* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al, 2019)](https://arxiv.org/abs/1910.10683)
* [Reformer: The Efficient Transformer (Kitaev et al, 2020)](https://arxiv.org/abs/2001.04451)
* [Attention Is All You Need (Vaswani et al, 2017)](https://arxiv.org/abs/1706.03762)
* [Deep contextualized word representations (Peters et al, 2018)](https://arxiv.org/pdf/1802.05365.pdf)
* [The Illustrated Transformer (Alammar, 2018)](http://jalammar.github.io/illustrated-transformer/)
* [The Illustrated GPT-2 (Visualizing Transformer Language Models) (Alammar, 2019)](http://jalammar.github.io/illustrated-gpt2/)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al, 2018)](https://arxiv.org/abs/1810.04805)
* [How GPT3 Works - Visualizations and Animations (Alammar, 2020)](http://jalammar.github.io/how-gpt3-works-visualizations-animations/)
