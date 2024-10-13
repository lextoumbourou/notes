---
title: "Learning Transferable Visual Models From Natural Language Supervision"
date: 2024-10-12 00:00
modified: 2024-10-12 00:00
status: draft
aliases:
- "Radford et al, 2020"
---

Notes from paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever (OpenAI).

## Overview

This paper introduces [CLIP](../permanent/contrastive-language-image-pretraining.md), a model for [Zero-Shot Learning](../permanent/zero-shot-learning.md), where the model is trained with a [Contrastive Loss](../permanent/contrastive-loss.md) objective, to associate correct images with correct text description. This allows the model to generalise to new tasks without fine-tuning on those tasks.

They benchmark n over 30 different existing computer vision datasets, across tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification.

The code and pre-trained model weights at https://github.com/OpenAI/CLIP.

![](../../../_media/Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision-fig-1.png)
*Figure 1. CLIP overview*

## Related Papers

**Unsupervised Pre-training in NLP**

Many papers have achieve good results doing Unsupervised Pre-training in NLP, including task objectives like [Masked Language Modelling](../../../permanent/masked-language-modelling.md):

* [Dai & Le, 2015: Semi-supervised Sequence Learning](papers/dai-le-2015-semi-supervised-sequence-learning.md)
* [Deep Contextualized Word Representations](../../../reference/deep-contextualized-word-representations.md)
* [Howard & Ruder, 2018: Universal Language Model Fine-tuning for Text Classification](reference/howard-ruder-2018-universal-language-model-fine-tuning-for-text-classification.md)
* [Improving Language Understanding by Generative Pre-Training](../../../reference/rimproving-language-understanding-by-generative-pre-training.md)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](../../../reference/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding.md)
* [Raffel et al., 2019: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](reference/raffel-et-al-2019-exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer.md)

**Text-to-text** models

They allowed task-agnostic architectures for zero-shot transfer to downstream datasets without fine-tuning:

* [McCann et al., 2018: The Natural Language Decathlon: Multitask Learning as Question Answering](reference/mccann-et-al-2018-the-natural-language-decathlon-multitask-learning-as-question-answering.md)
* [Radford et al., 2019: Language Models are Unsupervised Multitask Learners](reference/radford-et-al-2019-language-models-are-unsupervised-multitask-learners.md)
* [Raffel et al., 2019: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](reference/raffel-et-al-2019-exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer.md)

**General purpose LLMs**

And, powerful models like [GPT-3](../permanent/gpt-3.md) can do a bunch of tasks without specific training data:

* [Brown et al., 2020: Language Models are Few-Shot Learners](../permanent/brown-et-al-2020-language-models-are-few-shot-learners.md).

These results suggest that the aggregate supervision accessible to modern pre-training methods within web-scale collections of text surpasses that of high-quality crowd-labeled NLP datasets. This practice of pre-training on web-scale text datasets was not common practice, most models were trained on crowd-labelled datasets like ImageNet.

**Early Exploration of Content-Based Image Retrieval**

Over 20 years ago, Mori et al. (1999) explored ways to improve content-based image retrieval by training models to predict nouns and adjectives from text documents paired with images.

* [Mori et al., 1999: Improving Content-Based Image Retrieval with Text](reference/mori-et-al-1999-improving-content-based-image-retrieval-with-text.md)

**Learning Image Representations via Manifold Learning**

Quattoni et al. (2007) demonstrated that [[../../../permanent/manifold-learning]] in the weight space of classifiers could result in more data-efficient image representations by predicting words in captions associated with images.

* [Quattoni et al., 2007: Learning Efficient Image Representations via Manifold Learning](reference/quattoni-et-al-2007-learning-efficient-image-representations-via-manifold-learning.md)

**Multimodal Deep Boltzmann Machines**

Srivastava & Salakhutdinov (2012) explored deep representation learning by training multimodal Deep Boltzmann Machines on low-level image and text tag features.

* [Srivastava & Salakhutdinov, 2012: Deep Representation Learning with Multimodal Deep Boltzmann Machines](reference/srivastava-salakhutdinov-2012-deep-representation-learning-with-multimodal-deep-boltzmann-machines.md)

**Modernizing Image Representation Learning**

Joulin et al. (2016) modernized image representation learning by training CNNs to predict words in image captions, showing that these models learn useful image representations. They used the YFCC100M dataset (Thomee et al., 2016) to perform a bag-of-words multi-label classification task, and pre-trained AlexNet (Krizhevsky et al., 2012) on this data. They showed performance similar to ImageNet-based pre-training.

* [Joulin et al., 2016: Modernizing Image Representation Learning by Predicting Words in Captions](reference/joulin-et-al-2016-modernizing-image-representation-learning-by-predicting-words-in-captions.md)
* [Thomee et al., 2016: The YFCC100M Dataset](reference/thomee-et-al-2016-the-yfcc100m-dataset.md)
* [Krizhevsky et al., 2012: AlexNet](reference/krizhevsky-et-al-2012-alexnet.md)

**Extending to Phrase N-Grams**

Li et al. (2017) extended the approach to predict phrase n-grams, demonstrating the system's ability to zero-shot transfer to other image classification datasets by scoring target classes based on their learned visual n-grams and predicting the highest-scoring one.

* [Li et al., 2017: Predicting Phrase N-Grams for Zero-Shot Transfer](reference/li-et-al-2017-predicting-phrase-n-grams-for-zero-shot-transfer.md)

**Transformer-Based Approaches to Vision**

Recent studies, such as VirTex (Desai & Johnson, 2020), ICMLM (Bulent Sariyildiz et al., 2020), and ConVIRT (Zhang et al., 2020), have demonstrated the potential of transformer-based language modeling, masked language modeling, and contrastive objectives to learn image representations from text.

* [Desai & Johnson, 2020: VirTex - Vision-and-Language Pre-Training](reference/desai-johnson-2020-virtex-vision-and-language-pre-training.md)
* [Bulent Sariyildiz et al., 2020: ICMLM - Image Captioning with Masked Language Models](reference/bulent-sariyildiz-et-al-2020-icmlm-image-captioning-with-masked-language-models.md)
* [Zhang et al., 2020: Contrastive learning of medical visual representations from paired images and text](reference/zhang-et-al-2020-convirt-contrastive-learning-for-image-representation.md)

**Limitations of Natural Language Supervision for Image Representation**

While natural language supervision for image representation learning is an exciting proof of concept, it remains rare due to lower performance on common benchmarks. For example, Li et al. (2017) achieved only 11.5% accuracy on ImageNet in a zero-shot setting, much lower than the current state of the art of 88.4% accuracy (Xie et al., 2020). Even classic computer vision methods (Deng et al., 2012) achieve 50% accuracy, highlighting the gap.

* [Li et al., 2017: Predicting Phrase N-Grams for Zero-Shot Transfer](reference/li-et-al-2017-predicting-phrase-n-grams-for-zero-shot-transfer.md)
* [Xie et al., 2020: Current State of the Art in ImageNet Accuracy](reference/xie-et-al-2020-current-state-of-the-art-in-imagenet-accuracy.md)
* [Deng et al., 2012: Classic Computer Vision Methods for ImageNet](reference/deng-et-al-2012-classic-computer-vision-methods-for-imagenet.md)

**Narrowly Scoped Approaches with Weak Supervision**

More focused uses of weak supervision have significantly improved performance. Mahajan et al. (2018) demonstrated that predicting ImageNet-related hashtags from Instagram images is an effective pre-training task. When fine-tuned to ImageNet, this approach increased accuracy by over 5% and improved the overall state of the art.

* [Mahajan et al., 2018: Improving ImageNet Accuracy with Instagram Hashtags](reference/mahajan-et-al-2018-improving-imagenet-accuracy-with-instagram-hashtags.md)

**Broader Transfer Learning Gains with Noisy Label Pre-Training**

Kolesnikov et al. (2019) and Dosovitskiy et al. (2020) also demonstrated substantial improvements on broader transfer benchmarks by pre-training models to predict classes in the noisily labeled JFT-300M dataset.

* [Kolesnikov et al., 2019: Transfer Learning with Noisy Label Pre-Training](reference/kolesnikov-et-al-2019-transfer-learning-with-noisy-label-pre-training.md)
* [Dosovitskiy et al., 2020: Improving Transfer Benchmarks with Pre-Training](reference/dosovitskiy-et-al-2020-improving-transfer-benchmarks-with-pre-training.md)

A crucial difference between these weakly supervised models and recent explorations of learning image representations directly from natural language is scale.

While Mahajan et al. (2018) and Kolesnikov et al. (2019) trained their models for accelerator years on millions to billions of images, VirTex,
ICMLM, and ConVIRT trained for accelerator days on one to two hundred thousand images.

---

In this work, they close this gap and study the behaviors of image classifiers trained with natural language supervision at large scale.

Enabled by the large amounts of publicly available data of this form on the internet, they create a new dataset of 400 million (image, text) pairs and demonstrate that a simplified version of ConVIRT trained from scratch, which we call CLIP, for Contrastive Language-Image Pre-training, is an efficient method of learning from natural language supervision.

They study the scalability of CLIP by training a series of eight models spanning almost 2 orders of magnitude of compute and observe that transfer performance is a smoothly predictable function of compute.

They find that CLIP, like GPT family, learns to perform a wide set of tasks during pre-training including OCR, geo-localization, action recognition, and many others.

They also confirm these findings with linear-probe representation learning analysis and show that CLIP outperforms the best publicly available ImageNet model while also being more computationally efficient.

They find that zero-shot CLIP models are much more robust than equivalent accuracy supervised ImageNet models which suggests that zero-shot evaluation of task-agnostic models is much more representative of a model’s capability.
