---
title: "Neural Discrete Representation Learning"
date: 2024-10-16 00:00
modified: 2026-09-26 08:50
status: draft
---

Learning useful representations without supervision remains a key challenge in machine learning. In this paper, we propose a simple yet powerful generative model that learns such discrete representations. Our model, the Vector Quantised-Variational AutoEncoder ([VQ-VAE](../permanent/vq-vae.md)), differs from VAEs in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static. In order to learn a discrete latent representation, we incorporate ideas from vector quantisation (VQ). Using the VQ method allows the model to circumvent issues of “posterior collapse” (where the latents are ignored when they are paired with a powerful autoregressive decoder) typically observed in the VAE framework. Pairing these representations with an autoregressive prior, the model can generate high quality images, videos, and speech as well as doing high quality speaker conversion and unsupervised learning of phonemes, providing further evidence of the utility of the learnt representations.



![Title and abstract of Neural Discrete Representation Learning by van den Oord, Vinyals and Kavukcuoglu of DeepMind.](../_media/neural-discrete-representation-learning-title.png)

![Figure 1 from the paper: the VQ-VAE encoder output is mapped to its nearest codebook embedding before the decoder, and the gradient is copied straight through from decoder to encoder.](../_media/neural-discrete-representation-learning-fig1.png)
![Figure 2 from the paper: eight ImageNet images beside slightly blurrier VQ-VAE reconstructions from a 32x32 latent space with 512 codes.](../_media/neural-discrete-representation-learning-fig-2.png)

![Figure 6 from the paper: an original speech waveform, its reconstruction with the same speaker id, and its reconstruction with a different speaker id.](../_media/neural-discrete-representation-learning-fig6.png)

![Figure 4 from the paper: 128x128 frames of DeepMind Lab corridors sampled from a VQ-VAE with a PixelCNN prior.](../_media/neural-discrete-representation-learning-fig4.png)

![Section 3.3 of the paper: the prior over discrete latents is uniform during training, then a PixelCNN (images) or WaveNet (audio) is fitted over the latents for ancestral sampling.](../_media/neural-discrete-representation-learning-3-3-prior.png)