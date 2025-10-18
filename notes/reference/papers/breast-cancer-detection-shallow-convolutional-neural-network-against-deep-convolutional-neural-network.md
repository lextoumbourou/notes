---
title: "Breast cancer detection: Shallow convolutional neural network against deep convolutional neural network"
authors: ["Das, S.", "Saha, S.", "Roy, S.", "Chakraborty, C."]
year: 2023
journal: "Biomedical Signal Processing and Control"
doi: "10.1016/j.bspc.2023.105741"
citekey: dasBreastCancerCNN2023
status: draft
tags:
- BreastCancerDetection
- MedicalImaging
- CNN
date: 2025-10-18 00:00
aliases:
- "Das et al. (2023)"
---

> [!abstract]
> This study compares shallow and deep convolutional neural network (CNN) architectures for breast-cancer image classification. Using the BreakHis microscopic image dataset, the authors evaluate whether shallow architectures—requiring fewer parameters and computational resources—can achieve comparable accuracy to deeper networks. Results show that optimized shallow CNNs outperform several deeper counterparts in accuracy and training efficiency.

## Summary

### Background

* Breast cancer remains one of the leading causes of death among women.
* Deep learning methods, particularly CNNs, dominate computer-aided diagnosis.
* However, deeper networks demand higher computational cost, which limits clinical applicability.

### Objective

* To evaluate and compare **shallow vs. deep CNNs** for histopathological breast-cancer image classification.
* Determine if a shallow architecture can achieve comparable performance with reduced complexity.

### Dataset

* **BreakHis dataset** used: microscopic images of benign and malignant breast tumors at magnifications of 40×, 100×, 200×, 400×.
* Data split into training (70%) and testing (30%) sets.

### Methods

* Designed a **custom shallow CNN** with:
    * 3 convolutional layers
    * ReLU activations
    * Max-pooling
    * Dropout for regularization
    * Softmax output layer
* Compared against deeper CNN architectures (e.g., VGG16, ResNet-50, Inception V3).
* Evaluation metrics: accuracy, precision, recall, F1-score, and training time.

### Results

* **Shallow CNN** achieved ≈ 98.3 % accuracy at 400× magnification, surpassing deeper networks.
* Required significantly less computation time and memory.
* Deep models showed marginal accuracy gains (≤ 1 %) but much longer training durations.

### Discussion

* Shallow CNNs can generalize well on medical image datasets with limited size and variability.
* Over-parameterized deep models risk overfitting when training data are scarce.
* The study recommends architecture complexity proportional to dataset scale.

### Conclusion

* Optimised shallow CNNs can effectively detect breast cancer in histopathology images.
* They offer practical advantages for low-resource clinical and mobile screening systems.
* Future work: extend to multimodal data and hybrid deep–shallow ensembles.

### Key Takeaways

* **Shallow ≈ Deep in accuracy**, but **Shallow ≫ Deep in efficiency** for small medical datasets.
* **Dataset characteristics** and **computational constraints** should guide CNN architecture depth.
