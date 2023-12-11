---
title: Large-scale Contrastive Language-Audio Pre-training with Feature Fusion and Keyword-to-Caption Augmentation
date: 2023-12-06 00:00
modified: 2023-12-06 00:00
category: reference/papers
cover: /_media/cover-clap-paper.png
---

These are my notes from paper [Large-scale Contrastive Language-Audio Pre-training with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov

---

The main contributions of this paper are:

### 1. [Contrastive Language-Audio Pretraining](../../permanent/clap.md) (CLAP)

CLAP is a pre-training system for learning a text/audio latent space given pairs of examples.

It is trained using [Contrastive Loss](../../permanent/contrastive-loss.md). Hence: <span style="color: red;">Contrastive</span> <span style="color: blue;">Language-Audio</span> <span style="color: green;">Pre-training</span>.

They use a text-encoder and audio-encoder, to generate respective representations, then feed into a MLP layer to learn the shared latent space. For text-encoder, they use [RoBERTa](../../permanent/RoBERTa.md), for the audio-encoder, they use [HTS-AT](../../permanent/htsat.md), and provide details of other models evaluated.

They released the architecture, training code and series of weights trained on different subsets of their datasets [here](https://github.com/LAION-AI/CLAP) .

![](../../_media/paper-large-scale-contrastive-language-audio-retraining-with-feature-fusion-clap-overview.png)

#### 2. [Feature-Fusion](Feature-Fusion)

In the paper, they describe a system for dealing with variable-length, long audio (> 10 secs) called [Feature-Fusion](Feature-Fusion).

![](../../_media/paper-large-scale-contrastive-language-audio-retraining-with-feature-fusion-feature-fusion.png)

It combines the entire audio downsampled, alongside 10-second clips taken randomly throughout the audio. It's passed through a number of layers to get the final representation.

### 3. [Keyword-to-Caption Augmentation](../../permanent/keyword-to-caption-augmentation.md)

They also use [keytotext](https://github.com/gagan3012/keytotext) library to get more out of the keywords and labels in the [Audioset](Audioset) dataset.

### 4. [Laion Audio 630K](../../permanent/dataset-laion-audio-630K.md)

They also introduced Laion Audio 630k, which is a a large text/audio dataset scraped from various places.

For example:
* ![](../../_media/the-sound-of-a-siren.mov) "The sound of a siren"
* ![](../../_media/sound-of-wrestling-crowd.mov) "the sounds of wrestling crowd, mezzanine level, huge crowd, p.a., and loop."

---

### Tasks evaluated

* Text-to-Audio Retrieval: Achieves "superior performance"
* Zero-shot [Audio Classification](Audio%20Classification): State-of-the-art performance.
* Supervised [Zero-shot](Zero-shot): Comparable to performance.

---

## 1. Introduction

Most audio tasks in ML require annotated data. It's an ongoing challenge is unsupervised [Representation Learning](../../permanent/representation-learning.md) for audio.

[Contrastive Loss](../../permanent/contrastive-loss.md) a widely used solution for training on noisy internet data.

A particular implementation: [CLIP](../../permanent/contrastive-language-image-pretraining.md) learns the relationship between text and image by projecting into a shared [Latent Space](../../../../permanent/latent-space.md). It uses a ground-truth image-text pair as a positive sample and "left" as a negative.

It works because it's not constrained by data annotation, and has robustness in out-of-domain tests using [ImageNet](../../../../permanent/ImageNet.md). These weights are helpful for downstream tasks like text-to-image retrieval and text-guided captioning.

Vision, audio, and language have overlapping info. Text descriptions of events can be mapped to corresponding audio.

Text descriptions share a similar meaning that could be learned with the related audio to form an audio representation of cross-modal information.

Paired audio and text data are easy to collect.

Contrastive language-audio pre-training has been covered before.

[Text-to-audio retrieval via large-scale contrastive learning](Text-to-audio%20retrieval%20via%20large-scale%20contrastive%20learning)
* Pre-trained Audio Neural Network (PANN) as audio encoder.
* BERT as text encoder.
* Various loss functions to evaluate text-to-audio retrieval.

[Audio Retrieval with WavText5K and CLAP Training](../../permanent/audio-retrieval-with-wavtext5k-and-clap-training.md)
* Adds [HTS-AT](../../permanent/htsat.md) and [RoBERTa](../../permanent/RoBERTa.md) into encoder list to enhance performance.
* Uses representation in downstream task of audio classification.

Other studies focus on contrastive image-audio pre-training:
* [AudioCLIP: Extending CLIP to Image, Text and Audio](../../permanent/audioclip-extending-clip-to-image-text-and-audio.md)
* [Wav2CLIP: Learning Robust Audio Representations From CLIP](paper-wav2clip-learning-robust-audio-representations-from-clip.md)

All these previous studies do not show "full strength" of contrastive audio for language:
* Datasets "relatively small"
* Prior work hasn't done full investigation of selections and hyper-parameters.
* Can't accommodate varied audio lengths, particularly with [transformer](../../permanent/transformer.md)-based audio encoder.
* No analysis of representation in downstream task.

This paper makes the following contributions based on these concerns:

* Release [Laion Audio 630K](../../permanent/dataset-laion-audio-630K.md): largest public audio caption dataset of 633,526 audio-text pairs.
* Use keyword-to-caption model to augment labels of AudioSet into corresponding captions.
* Make a pipeline of contrastive language-audio pre-training:
    * Select 2 audio encoders and 3 text encoders for testing.
    * Use feature fusion mechanisms to improve performance and to support variable length inputs.
* Comprehensive experiments:
    * text-to-audio retrieval task.
    * zero-shot classification.
    * supervised audio classification.

Achieve SOTA in text-to-audio retrieval and audio classification (even compared to supervised models).

## 2. Laion-Audio-630K and Training Dataset

#### [Laion Audio 630K](../../permanent/dataset-laion-audio-630K.md)

633,526 text/audio pairs.

Total duration of 4,325.39 hours.

Contains:
* human activities
* natural sounds
* audio effects

Data from 8 publicly available websites.

#### Training datasets

They use 3 different datasets to train, testing model performance on different sizes and types of dataset:

* [AudioCaps](AudioCaps) + [Clotho](Clotho) (AC+CL)
    * 55k audio-text pairs.
* [Laion Audio 630K](../../permanent/dataset-laion-audio-630K.md)
    * 630k audio-text pairs.
* [Audioset](Audioset)
    * 1.9 million audio samples with only labels available for each sample.

Exclude overlapping data in evaluation sets.


#### Dataset Format and Preprocessing

All audio converted to mono.

48kHz in FLAC format.

Data with only tags or labels, expand labels into captions using template:

* sound of `label-1`, `label-2`, ..., and `label-n`

Total audio samples: 2.5 million.

## 3. Model Architecture

### 3.1 Contrastive Language-Audio Pre-training

[Contrastive Language-Audio Pretraining](../../permanent/clap.md)

The proposed architecture.

![](../../../../_media/large-scale-contrastive-language-audio-retraining-with-feature-fusion-fig-1.png)

##### Notation

$X^{a}_i$ = audio example i
$X^{t}_{i}$ = text example i
$X^{a}_{i}, X^{t}_{i}$ = audio-text pair $i$.

Like [CLIP](../../permanent/contrastive-language-image-pretraining.md) has 2 encoders:
$E^{a}_{i}$ = audio [embedding](../../permanent/embedding.md) i, obtained using audio encoder: $faudio(.)$
$E^{t}_{i}$ = text [embedding](../../permanent/embedding.md) , obtained using text encoder: $ftext(.)$

Uses [Projection](Projection) layers:

$E^{a}_{t} = \text{MLPaudio}(\text{faudio}(X^{a}_{i}))$
$E^{t}_{i} = \text{MLPtext}(\text{ftext}(X^{t}_{i}))$

Where audio/text projectiction layer is a 2-layer multilayer perceptron (MLP) with ReLU as activation function to map encoder outputs into sampe dimensions (i.e., Eai, Et i ∈ RD)

Trained with contrastive learning paradigm between the audio and text embeddings in pair, following the same loss function in [1]:
$L = \frac{1}{2N} \Sigma^{N}_{i=1} \ ( \log  \frac{\exp{E^{a}_{i} \cdot E^{t}_{i} / r}}{\Sigma^{N}_{j=1} \exp(E^{a}_{i} \cdot E^{t}_{j} / r)} + \log \frac{\exp{E^{t}_{i} \cdot E^{a}_{i} / r}}{\Sigma^{N}_{j=1} \exp(E^{t}_{i} \cdot E^{a}_{j} / r)} )$

Where:
* *$r$ is a learnable temperature param for scaling the loss.
* 2 logarithmic terms consider either audio-to-text logits or text-to-audio logits.
* N batch size.
* After training, embeddings are useful for different tasks.


### 3.2 Downstream Tasks in Inference Stage

#### Text-to-Audio Retrieval

Target audio embedding $E^{a}_{p}$ can find the nearest text embedding $E^{t}_{q}$ among $M$ texts $E^{t} = \{E^{t}_{1}, ...E^{t}_{M}\}$ using cosine similarity function.

#### Zero-shot audio classification

For M audio classes $C = {c_1, ..., C_m}$ we can construct M prompt texts $X^{t} = \{X^{t}_{1}, ...., X^{t}_{M}\}$ using a string like: "the sound of `class-name`".

One advantage of using the contrastive language-audio pre-training is that the categories of audio are unrestricted (i.e., zero-shot) since the model can convert the classification task into the text-to-audio retrieval task.
* Supervised Audio Classification
    * After training the model, for a given audio $X^{a}_{p}$ its embeddings $E^{a}_{p}$ can be further mapping to fixed-category classification task by adding a projection layer at the back and finetuning (i.e., the non-zero-shot setting).

### 3.3 Audio Encoders and Text Encoders

Try two models for audio encoder:

[PANNs](../../../../permanent/PANNs.md)

CNN-based audio classification model with 7 downsampling CNN blocks and 7 upsampling blocks.

[HTS-AT](../../permanent/htsat.md)

HTSAT is a transformer-based model with 4 groups of swintransformer blocks, which achieves SOTAs on three audio classification datasets.

For both, they use the 2nd last layer's output.

PANNs: 2048 and HTSAT: 768

For text encoder, they try:
* CLIP transformer
    * Output dimension 512
* BERT
    * Output dimension 768
* RoBERTa
    * 768

Use 2-layer MLPs with [ReLU](../../../../permanent/ReLU.md) activation

Map audio and text into 512 dimensions, for contrastive learning.

### 3.4 Feature Fusion for Variable-Length Audio

[Feature-Fusion](Feature-Fusion)

Problem: audio is natural variable length, until image data which we can resize to "unified resolution" (ie 224x224).
Possible solution: average per-frame or per-chunk audio embeddings as output. However, it's computationally inefficient on long audio.

CLAP approach: Feature Fusion (see Fig. 1 above)

Train on different lengths of audio inputs in constant computation time by combining both coarsely global and randomly sampled local information.

For an audio in T seconds and a fixed chunk duration d = 10 seconds:

$T < d$: repeat the input, then pad with zero values. For example, 3-second input will be repeated 3 x 3 = 9 second and padded with 1-second zero values.

$T > d$:

* first downsample the input from T to d-second (10 seconds) as global input.
    * These are the global inputs.
* then randomly slice three d-second clips: in front 1/3, then middle 1/3 and back 1/3 of the input.
    * These are the local inputs.
* Send these $4 \ x \ d$ inputs to the audio encoder to get the initial features.
    * I believe this is a "Mel-FilterBank"
* Also, send the 3 local inputs to 2D Conv layer with 3-stride in time axis, to convert to one feature.
* Now fuse the local feature $X^{a}_{local}$ and the global feature $X^{a}_{global}$ : $X^{x}_{fusion} = \lambda X^{a}_{global}.+ (1 - \alpha)X^{a}_{local}$
* Where $α = fAF F (Xaglobal, Xa local)$ is a factor obtained by attention feature fusion (AFF) [21], a two-branch CNN model for learning the fusion factor of two inputs.

Code from the repo:

```python
mel = get_mel(audio_data, audio_cfg)

ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    
if len(ranges[1]) == 0:
    # if the audio is too short, we just use the first chunk
    ranges[1] = [0]
if len(ranges[2]) == 0:
    # if the audio is too short, we just use the first chunk
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

Comparing with the "slice & vote" method, the feature fusion also saves the training time as we only process audio slices in the first few layers.

### 3.5 Keyword-to-Caption Augmentation

[Keyword-to-Caption Augmentation](../../permanent/keyword-to-caption-augmentation.md)

Some datasets contain only labels or tags of audio.

They use a pre-trained language model to make captions on the keywords.

The specific details aren't in the paper but mention in [this](https://github.com/LAION-AI/CLAP/issues/58#issuecomment-1377489927) GitHub issue.

 ![](../../../../_media/large-scale-contrastive-language-audio-retraining-with-feature-fusion-gt-issue.png)
Also do a [De-bias](De-bias) the output sentence in post-processing.

Replace woman and man with person and others mentioned in appendix.

## 4. Experiments

3 experiments conducted:

* Try different audio and text encoders to get best baseline model.
* Train model on various dataset size, with feature fusion and keyword-to-caption augmentation to verify efficacy of proposed methods.

These 2 evaluated via recall and mean average precision (mAP) on audio-to-text and text-to-audio.

Lastly, they do zero-shot and supervised audio classification experiments to evaluate the generalisation ability to the downstream tasks.

### 4.1. Hyperparameters and Training Details

Use [AudioCaps](AudioCaps), [Clotho](Clotho), [LAIONAudio-630K](LAIONAudio-630K), also [AudioSet](AudioSet) by keyword-to-caption augmentation, to train model.

For the audio:
* 10-second input length
* 480 hop size
* 1024 window size
* 64 mel-bins to compute STFTs and mel-spectrograms

So, each input sent to the audio encoder is of the shape (T = 1024, F = 64).

For the text:
* tokenize the text with a maximum token length of 77

When training the model without the feature fusion, the audio longer than 10-second will be randomly chunked to a 10-second segment.

During training, we use the Adam [23] optimizer with β1 = 0.99, β2 = 0.9 with a warm-up [24] and cosine learning rate decay at a basic learning rate of 10−4

We train the model using a batch size of 768 on AudioCaps+Clotho dataset, 2304 on training dataset containing LAION-Audio-630K, and 4608 on training dataset containing AudioSet. We train the model for 45 epochs.

### 4.2. Text-to-Audio Retrieval

Experiments to pick best audio and text encoder for retrieval task.

Combine 2 audio encoders with 3 text encoders, where both are loaded from pre-trained checkpoints.

In this experiment, train on [AudioCaps](AudioCaps) and [Clotho](Clotho) datasets, and report the best mAP@10 on audio-to-text (A→T) and text-to-audio (T→A) perspectives

Results:

* Audio encoder: [HTS-AT](../../permanent/htsat.md) better than [PANNs](../../../../permanent/PANNs.md) combined with RoBERTa or BERT.
* Text encoder: RoBERTa beats BERT. CLIP transformer worst.
    ![](../../../../_media/large-scale-contrastive-language-audio-retraining-with-feature-fusion-table-2.png)


#### Dataset Scale

Apply HTSAT-RoBERTa as our best model setting to conduct the text-to-audio retrieval experiments as a comprehensive evaluation in Table 3.

Same metrics in [7, 8] to compute recall scores at different ranks in this task.

In the training set, we gradually increase the scale of the dataset

We find that scaling up the dataset from "AudioCaps + Clotho" to "LA." does not improve the result on AudioCaps evaluation set but gets better performance on Clotho evaluation set, which is similar to the comparison between MMT [7] and CLAP-HTSAT [5].

One reason is that AudioCaps contains audios similar to AudioSet on which the audio encoder's loaded checkpoint is pretrained.

When the model receives more data from other sources, it increases its generalization but moves the distribution out of AudioSet data.

Therefore, the performance on AudioCaps drops but that on Clotho increases a lot, demonstrating a trade-off of the model to keep the performance among different types of audios.

Keyword-to-Caption and Feature Fusion When adding the feature fusion mechanism and keyword-to-caption augmentation to the
model, we can observe that either of them improves the performance

The feature fusion is effective especially in Clotho dataset because it contains longer audio data (> 10-second).

When we add AudioSet into the training set with either template prompting or keyword-tocaption augmentation, we can see the performance increases again on AudioCaps while decreases on Clotho.

This further confirms the trade-off performance between AudioCaps and Clotho datasets mentioned above

And the keyword-to-caption augmentation does bring in better performance than the simple template text prompting method on most metrics

As the result, our best model outperforms previous methods on most metrics (mainly R@1=36.7% on AudioCaps and R@1=18.2% on Clotho) in the text-to-audio retrieval tasks.

We show that training on large-scale datasets (LAION-Audio-630K and AudioSet with keyword-to caption augmentation), and feature fusion can effectively improve model performance.

### 4.3. Zero-shot and Supervised Audio Classification

Zero-shot Audio Classification To study the model generalization and robustness, we conduct zero-shot audio classification experiments on three top-performing models in previous experiments.

We evaluate models on three audio classification dataset, namely ESC50 [27], VGGSound [28], and Urbansound8K (US8K) [29]. We use top-1 accuracy as the metric

They classify audio by performing audio-to-text retrieval with each text corresponds to the text prompt converted from class label via "This a sound of label.".

We noticed a dataset overlap between our training data and the zero-shot dataset we are evaluating on. We excluded all the overlap samples and perform zero-shot evaluation on the whole remaining dataset

Supervised Audio Classification We perform supervised audio classification by fine-tuning the audio encoder on FSD50K [30] and VGGSounddatasets. We do not conduct this experiment on ESC50 and Urbansound8K because the potential data leakage issue in those dataset will makes the results incomparable with the previous methods. Specially, mAP is used as the metric to evaluate FSD50K.

As shown in the in Table 4, our models achieves new SoTAs of zero-shot audio classification across all three datasets, demonstrating the high generalization ability of our model to unseen data. Keyword-to-Caption augmentation increases the performance of VGGsound and US8K a lot as it adds more text captions to "enrich" the text embedding space.

Feature-fusion not only enables the model to handle variable-length input, but also achieves better performance than previous models

Supervised audio classification result: outperforms the current state-of-the-art on VGGSound dataset close to state-of-the-art on FSD50K dataset.

This tells us we're learning an effective audio representation.

## 5. Future Work

They want to collect an even larger dataset.

Apply representations to more downstream tasks like audio synthesis and separation.
