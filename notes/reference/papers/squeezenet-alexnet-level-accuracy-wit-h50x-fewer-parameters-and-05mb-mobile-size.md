---
title: "SqueezeNet: AlexNet-level accuracy with 50× fewer parameters and < 0.5 MB model size"
authors: ["Iandola, F.N.", "Han, S.", "Moskewicz, M.W.", "Ashraf, K.", "Dally, W.J.", "Keutzer, K."]
year: 2016
doi: "10.48550/arXiv.1602.07360"
citekey: iandolaSqueezeNet2016
tags:
- CNN
- ModelCompression
- DeepLearning
- ArchitectureDesign
- ParameterEfficiency
status: draft
aliases:
- "Iandola et al. (2016)"
---

> [!abstract]
> SqueezeNet achieves AlexNet-level ImageNet accuracy while reducing parameters > 50× through architectural design, not post-training compression. It introduces the **Fire module**, combines **1×1 and 3×3 filters** to reduce computation, and employs **late down-sampling** to preserve activation resolution.

## Key Architectural Changes for Parameter Reduction

### 1. **Fire Module: “Squeeze → Expand” design**
* Core building block combining two stages:
  * **Squeeze layer:** uses only 1×1 convolutions to reduce the number of input channels → minimises parameters feeding the next layer.
  * **Expand layer:** a mix of 1×1 and 3×3 filters applied to the squeezed output.
* Parameter formula highlights the saving:  
  fewer 3×3 filters → fewer weights (each 3×3 has 9× more parameters than 1×1).

### 2. **Ratio control hyperparameters**
* Introduced tunables:
  * **squeeze_ratio (SR):** proportion of channels in the squeeze layer relative to the expand layer (default = 0.125).
  * **percent_3x3:** fraction of expand filters that are 3×3 (typically = 0.5).  
  These allow fine-grained balance between accuracy and parameter count.

### 3. **Replace large filters with 1×1 filters**
* Following Network-in-Network ideas, 1×1 convolutions replace most 3×3 and 5×5 kernels, drastically cutting parameters and memory bandwidth.

### 4. **Delayed down-sampling**
* Pooling and stride > 1 occur **later** in the network than in AlexNet.  
  → Larger activation maps for more layers, yielding higher accuracy at similar cost.

### 5. **Model compression compatibility**
* Further 10× reduction achieved using **Deep Compression** (Han et al., 2016): pruning + quantization + Huffman coding → < 0.5 MB final model.

![squeeze-net-fig1.png](../../../../_media/squeeze-net-fig1.png)

## Comparison to Typical CNN (AlexNet)

| Feature | AlexNet | SqueezeNet |
|:--|:--|:--|
| Conv kernel size | Many 3×3 and 5×5 | Mostly 1×1 (+ few 3×3 in expand) |
| Filters per layer | Hundreds | 16–512 per Fire module |
| Down-sampling | Early (layer 2) | Late (after Fire 8) |
| Parameters | ≈ 60 M | ≈ 1.2 M (50× fewer) |
| Accuracy (top-1 ImageNet) | 57.2 % | 57.5 % |
| Model size | ≈ 240 MB | ≈ 4.8 MB (≤ 0.5 MB after compression) |

## Key Insights

* **Parameter efficiency via architectural design**, not heavy regularization or compression after training.  
* Demonstrates **micro-architecture (Fire module)** and **macro-architecture (down-sampling schedule)** jointly control accuracy–size trade-off.  
* Became a foundation for later lightweight CNNs (MobileNet, ShuffleNet).

---

**Reference:**  
* Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). *SqueezeNet: AlexNet-level accuracy with 50× fewer parameters and < 0.5 MB model size.* arXiv:1602.07360.  