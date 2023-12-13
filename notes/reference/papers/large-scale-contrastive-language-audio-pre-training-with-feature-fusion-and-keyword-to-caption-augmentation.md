---
title: "Large-scale Contrastive Language-Audio Pre-training with Feature Fusion and Keyword-to-Caption Augmentation"
date: 2023-12-13 00:00
modified: 2023-12-13 00:00
category: reference/papers
cover: /_media/cover-clap-paper.png
summary: Notes from paper [Large-scale Contrastive Language-Audio Pre-training with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov
hide_cover_in_article: true
---

These are my notes from the paper [Large-scale Contrastive Language-Audio Pre-training with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov

## Overview

This paper concerns learning a text/audio latent space using constrastive learning. Like [CLIP](https://openai.com/research/clip) for audio. It has four main contributions:

### 1. Contrastive Language-Audio Pre-training (CLAP)

[CLAP](../../permanent/clap.md) is a pre-training system for learning a text/audio latent space given pairs of examples.

It is trained using [Contrastive Loss](../../permanent/contrastive-loss.md). Hence: <span style="color: red;">Contrastive</span> <span style="color: blue;">Language-Audio</span> <span style="color: green;">Pre-training</span>.

They use a text-encoder and audio-encoder to generate respective representations, then feed into an MLP layer to learn a shared latent space. For the text-encoder, they use [RoBERTa](../../permanent/RoBERTa.md), and for the audio-encoder, they use [HTS-AT](../../permanent/hts-at.md) and provide details of other models evaluated.

They released the architecture, training code and series of weights trained on different subsets of their datasets [here](https://github.com/LAION-AI/CLAP).

![](../../_media/paper-large-scale-contrastive-language-audio-retraining-with-feature-fusion-clap-overview.png)

### 2. [Feature-Fusion](../../permanent/feature-fusion.md)

A technique for dealing with variable-length, long audio (> 10 secs). It combines the entire audio downsampled, alongside 10-second clips taken randomly throughout the audio. They pass through several layers to get the final representation.

![Feature-fusion](../../_media/paper-large-scale-contrastive-language-audio-retraining-with-feature-fusion-feature-fusion.png)

### 3. [Keyword-to-Caption Augmentation](../../permanent/keyword-to-caption-augmentation.md)

Since the [Audioset](Audioset) dataset only has keywords and labels paired with audio, they use a T5 model provided by the [keytotext](https://github.com/gagan3012/keytotext) library to create captions from these keywords.

They also add a de-biasing step to convert references to "woman" or "man" to "person".

Here are some examples of the keywords and the generated captions, followed by their debiased versions.

![](../../_media/large-scale-contrastive-language-audio-pre-training-with-feature-fusion-and-keyword-to-caption-augmentation-captions-debias.png)

### 4. [Laion Audio 630K](../../permanent/laion-audio-630K.md)

Finally, Laion Audio 630k was released: a large text/audio dataset scraped from the internet. They collect 633 526 text/audio pairs, amounting to 4,325.39 hours of audio.

Contains:

* human activities
* natural sounds
* audio effects

#### Examples

##### "The sound of a siren"

<audio controls>
  <source src="/_media/the-sound-of-a-siren.mov" type="audio/mpeg">
</audio>

##### "The sounds of wrestling crowd, mezzanine level, huge crowd, p.a., and loop."

<audio controls>
  <source src="/_media/sound-of-wrestling-crowd.mov" type="audio/mpeg">
</audio>

## Evaluated Task Summary

* Text-to-Audio Retrieval: achieves "superior performance."
* Zero-shot Audio Classification: State-of-the-art performance.
* Supervised Audio Classification: comparable performance.

## Related work

The paper's primary goal is to test large-scale unsupervised [Representation Learning](../../permanent/representation-learning.md) for audio. One of the most successful implementations of this, [Contrastive Language-Image Pre-training (CLIP)](../../permanent/contrastive-language-image-pretraining.md), learns the relationship between text and image by projecting into a shared latent space. CLIP uses [Contrastive Loss](../../permanent/contrastive-loss.md), which seems to be good at these problems involving "noisy internet data".

It uses a ground-truth image-text pair as a positive sample and "left" as a negative. It works well because it's not constrained by data annotation. The representation is helpful for downstream tasks like text-to-image retrieval and text-guided captioning.

Vision, audio, and language have overlapping info; text descriptions of events can be mapped to corresponding audio. Text descriptions share a similar meaning that could be learned with the related audio to form an audio representation of cross-modal information. Also, paired audio and text data is relatively easy to collect.

Contrastive language-audio pre-training has been covered before:

* [Text-to-audio retrieval via large-scale contrastive learning](Text-to-audio%20retrieval%20via%20large-scale%20contrastive%20learning)
    * Pre-trained Audio Neural Network (PANN) as the audio encoder.
    * BERT as text encoder.
    * Various loss functions to evaluate text-to-audio retrieval.

* [Audio Retrieval with WavText5K and CLAP Training](../../permanent/audio-retrieval-with-wavtext5k-and-clap-training.md)
    * Adds [HTS-AT](../../permanent/hts-at.md) and [RoBERTa](../../permanent/RoBERTa.md) into encoder list to enhance performance.
    * Uses representation in the downstream task of audio classification.

* Other studies focus on contrastive image-audio pre-training:
    * [AudioCLIP: Extending CLIP to Image, Text and Audio](../../permanent/audioclip-extending-clip-to-image-text-and-audio.md)
    * [Wav2CLIP: Learning Robust Audio Representations From CLIP](paper-wav2clip-learning-robust-audio-representations-from-clip.md)

However, none of these previous studies shows contrastive audio's "full strength" for language. They have the following limitations:

* Datasets "relatively small."
* Prior work has yet to investigate selections and hyper-parameters thoroughly.
* Can't accommodate varied audio lengths, particularly with [transformer](../../permanent/transformer.md)-based audio encoder.
* No analysis of representation in the downstream task.

## Training Dataset

On top of [Laion Audio 630K](../../permanent/laion-audio-630K.md), they incorporate two other academic datasets in their training experiments:

* [AudioCaps](AudioCaps) + [Clotho](Clotho) (AC+CL)
    * 55k audio-text pairs.
* [Audioset](Audioset)
    * 1.9 million audio samples with only labels available for each sample.

They also exclude overlapping data in evaluation sets.

### Dataset Format and Preprocessing

All audio is converted to mono.

48kHz in FLAC format.

Data with only tags or labels, expand labels into captions using the template: the sound of `label-1`, `label-2`, ..., and `label-n`

Ultimately, they had a total collection of 2.5 million audio samples.

## Model Architecture

Notation:

* $X^{a}_i$ = audio example i
* $X^{t}_{i}$ = text example i
* $X^{a}_{i}, X^{t}_{i}$ = audio-text pair $i$.

Like [CLIP](../../permanent/contrastive-language-image-pretraining.md) has 2 encoders:

* $E^{a}_{i}$ = audio [embedding](../../permanent/embedding.md) i, obtained using audio encoder: $faudio(.)$
* $E^{t}_{i}$ = text [embedding](../../permanent/embedding.md) , obtained using text encoder: $ftext(.)$

And uses [Projection](Projection) layers:

* $E^{a}_{t} = \text{MLPaudio}(\text{faudio}(X^{a}_{i}))$
* $E^{t}_{i} = \text{MLPtext}(\text{ftext}(X^{t}_{i}))$

The audio/text projection layer is a 2-layer multilayer perceptron (MLP) with ReLU as an activation function to map encoder outputs into the exact dimensions.

Trained with contrastive learning paradigm between the audio and text embeddings in pairs, using this loss function:

$L = \frac{1}{2N} \Sigma^{N}_{i=1} \ ( \log  \frac{\exp{E^{a}_{i} \cdot E^{t}_{i} / r}}{\Sigma^{N}_{j=1} \exp(E^{a}_{i} \cdot E^{t}_{j} / r)} + \log \frac{\exp{E^{t}_{i} \cdot E^{a}_{i} / r}}{\Sigma^{N}_{j=1} \exp(E^{t}_{i} \cdot E^{a}_{j} / r)} )$

Where:

* $r$ = learnable temperature parameter for scaling the loss.
* 2 logarithmic terms consider either audio-to-text logits or text-to-audio logits.
* N = batch size.

After training, embeddings are useful for different tasks, including:

* Text-to-Audio Retrieval
    * Target audio embedding $E^{a}_{p}$ can find the nearest text embedding $E^{t}_{q}$ among $M$ texts $E^{t} = \{E^{t}_{1}, ...E^{t}_{M}\}$ using cosine similarity function.
* Zero-shot audio classification
    * For M audio classes $C = {c_1, ..., C_m}$ we can construct M prompt texts $X^{t} = \{X^{t}_{1}, ...., X^{t}_{M}\}$ using a string like: "the sound of `class-name`".
    * A key advantage of contrastive pre-training is that the categories can be anything as long as the information is in training data.
* Supervised Audio Classification
    * Use a given audio $X^{a}_{p}$ embeddings $E^{a}_{p}$ and fine-tune on a classification task.

### Audio Encoders and Text Encoders

Try two models for audio encoder:

* [PANNs](../../../../permanent/PANNs.md) - a CNN-based audio classification model with seven downsampling CNN blocks and seven upsampling blocks.
* [HTS-AT](../../permanent/hts-at.md) - a transformer-based model with four Swintransformer blocks, achieving SOTAs on three audio classification datasets.

For both, they use the 2nd last layer's output. PANNs: 2048-sized embedding and HTSAT: a 768-sized embedding

For text encoder, they try:

* CLIP transformer (output dimension 512)
* BERT (output dimension 768)
* Roberta (output dimension 768)

Use 2-layer MLPs with [ReLU](../../../../permanent/ReLU.md) activation, which maps audio and text into 512 dimensions for contrastive learning.

### Feature Fusion for Variable-Length Audio

Audio is a natural variable length, unlike image data, which we can resize to "unified resolution" (i.e. 224x224).

A common approach is to average per-frame or per-chunk audio embeddings as output. However, it could be more computationally efficient on long audio.

[Feature-Fusion](../../permanent/feature-fusion.md):

Train on different lengths of audio inputs in constant computation time by combining coarsely global and randomly sampled local information.

For an audio in T seconds and a fixed chunk duration d = 10 seconds:

$T < d$: repeat the input, then pad with zero values. For example, 3-second input will be repeated 3 x 3 = 9 seconds and padded with 1-second zero values.

$T > d$:

* First, downsample the input from T to d-second (10 seconds) as global input. These are the global inputs.
* Then, randomly slice three d-second clips: in front 1/3, then middle 1/3 and back 1/3 of the input. These are the local inputs.
* Send these $4 \ x \ d$ inputs to the mel encoder to get the initial features.
* send the three local inputs to the 2D Conv layer with a 3-stride in the time axis to convert to one feature.
* Now fuse the local feature $X^{a}_{local}$ and the global feature $X^{a}_{global}$: $X^{x}_{fusion} = \lambda X^{a}_{global}.+ (1 - \alpha)X^{a}_{local}$, where $α = fAF F (Xaglobal, Xa local)$ is a factor obtained by attention feature fusion (AFF) [21], a two-branch CNN model for learning the fusion factor of two inputs.

Code from the repo:

```python
mel = get_mel(audio_data, audio_cfg)

ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    
if len(ranges[1]) == 0:
    # if the audio is too short, we use the first chunk
    ranges[1] = [0]
if len(ranges[2]) == 0:
    # If the audio is too short, we use the first chunk
    ranges[2] = [0]

# randomly choose index for each part
idx_front = np.random.choice(ranges[0])
idx_middle = np.random.choice(ranges[1])
idx_back = np.random.choice(ranges[2])

# select mel parts for local inputs
mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

# shrink the mel to create global inputs
mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, audio_cfg['mel_bins']])(mel[None])[0]

# stack
mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
```

Compared with the "slice & vote" method, the feature fusion saves training time as we only process audio slices in the first few layers.

### Keyword-to-Caption Augmentation

See the Main Contributions section.

## Experiments

Three experiments were conducted:

They evaluated the first two using recall and mean average precision (mAP) on audio-to-text and text-to-audio.

* Try different audio and text encoders to get the best baseline model.
* Train model on various dataset sizes, with feature fusion and keyword-to-caption augmentation to verify the efficacy of proposed methods.

Lastly, they do zero-shot and supervised audio classification experiments to evaluate the generalisation ability of the downstream tasks.

### Hyper-parameters and Training Details

Use [AudioCaps](AudioCaps), [Clotho](Clotho), [LAIONAudio-630K](LAIONAudio-630K), also [AudioSet](AudioSet) by keyword-to-caption augmentation, to train model.

For the audio:

* 10-second input length
* 480 hop size
* 1024 window size
* 64 mel-bins to compute STFTs and mel-spectrograms

So, each input sent to the audio encoder is of the shape (T = 1024, F = 64).

For the text:

* tokenise the text with a maximum token length of 77

When training the model without the feature fusion, the audio longer than 10 seconds will be randomly chunked into a 10-second segment.

##### Optimiser

Use the [[Adam]] optimiser with β1 = 0.99, β2 = 0.9 with a warm-up and cosine learning rate decay at a basic learning rate of 10−4.

##### Batch sizes

* Batch size of 768 on the AudioCaps+Clotho dataset
* 2304 on the training dataset containing LAION-Audio-630K
* 4608 on the training dataset containing the AudioSet

##### Epochs

We train the model for 45 epochs.

### Text-to-Audio Retrieval

They experiment to find the best audio and text encoder for retrieval tasks.

Combine two audio encoders with three text encoders loaded from pre-trained checkpoints.

In this experiment, train on [AudioCaps](AudioCaps) and [Clotho](Clotho) datasets, and report the best mAP@10 on audio-to-text (A→T) and text-to-audio (T→A) perspectives

Results:

* Audio encoder: [HTS-AT](../../permanent/hts-at.md) better than [PANNs](../../../../permanent/PANNs.md) combined with RoBERTa or BERT.
* Text encoder: RoBERTa beats BERT. CLIP transformer worst.

![](../../_media/large-scale-contrastive-language-audio-retraining-with-feature-fusion-table-2.png)
#### Dataset Scale

Now, they use the HTSAT-RoBERTa model to perform text-to-audio retrieval experiments across the datasets.

This task uses the same metrics to compute recall scores at different ranks.

In the training set, they gradually increase the scale of the dataset.

They find that scaling up the dataset from "AudioCaps + Clotho" to "LA." does not improve the result on the AudioCaps evaluation set but gets better performance on the Clotho evaluation set.

One reason is that AudioCaps contains audio similar to AudioSet, on which the audio encoder's loaded checkpoint is pre-trained.

When the model receives more data from other sources, it increases its generalisation but moves the distribution out of AudioSet data.

Therefore, the performance on AudioCaps drops, but that on Clotho increases a lot, demonstrating a trade-off of the model to keep the performance among different types of audio.

#### Keyword-to-Caption and Feature Fusion

They observe performance improvements when adding the feature fusion mechanism and keyword-to-caption augmentation to the model.

Feature fusion is effective, especially in the Clotho dataset, because it contains longer audio data (> 10 seconds).

When they add AudioSet into the training set with either template prompting or keyword-to-caption augmentation, we can see the performance increases again on AudioCaps while decreasing on Clotho.

This result further confirms the trade-off performance between AudioCaps and Clotho datasets mentioned above.

Also, keyword-to-caption augmentation works better than the template text prompting method on most metrics.

As a result, our best model outperforms previous methods on most metrics (mainly R@1=36.7% on AudioCaps and R@1=18.2% on Clotho) in the text-to-audio retrieval tasks.

We show that training on large-scale datasets (LAION-Audio-630K and AudioSet with keyword-to-caption augmentation) and feature fusion can improve model performance.

![](../../_media/large-scale-contrastive-language-audio-pre-training-with-feature-fusion-and-keyword-to-caption-augmentation-table-3.png)

### Zero-shot and Supervised Audio Classification

#### Zero-shot Audio Classification

To study the model generalisation and robustness, they conducted zero-shot audio classification experiments on three top-performing models in previous experiments.

They evaluate models on three audio classification datasets, namely [[ESC50]], [[VGGSound]], and [[Urbansound8K]]. They use top-1 accuracy as the metric.

They classify audio by performing audio-to-text retrieval with each text corresponding to the text prompt converted from class label via "This a sound of the *label*.".

There's some dataset overlap between the training data and the zero-shot dataset. They excluded all the overlap samples and performed a zero-shot evaluation on the remaining dataset.

![](../../_media/large-scale-contrastive-language-audio-pre-training-with-feature-fusion-and-keyword-to-caption-augmentation-zeroshot.png)

#### Supervised Audio Classification

They perform supervised audio classification by finetuning the audio encoder on FSD50K [30] and VGGSound datasets. We do not conduct this experiment on ESC50 and Urbansound8K because the potential data leakage issue in that dataset will make the results incomparable with the previous methods. In particular, mAP is used as the metric to evaluate FSD50K.

As shown in Table 4, our models achieve new SoTAs of zero-shot audio classification across all three datasets, demonstrating the high generalisation ability of our model to unseen data. Keyword-to-caption augmentation dramatically increases the performance of VGGsound and US8K by adding more text captions to "enrich" the text embedding space.

Feature-fusion enables the model to handle variable-length input and performs better than previous models.

Supervised audio classification result: outperforms the current state-of-the-art on the VGGSound dataset close to the state-of-the-art on the FSD50K dataset. Therefore, this tells us they've learned an effective audio representation.

![](../../_media/large-scale-contrastive-language-audio-pre-training-with-feature-fusion-and-keyword-to-caption-augmentation-supervised.png)

![](../../_media/large-scale-contrastive-language-audio-pre-training-with-feature-fusion-and-keyword-to-caption-augmentation-table-4.png)

## Future Work

They want to collect an even larger dataset.

Apply representations to more downstream tasks like audio synthesis and separation.