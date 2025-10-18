---
title: "Mammography with Deep Learning for Breast Cancer Detection"
date: 2025-10-14 00:00
modified: 2025-10-14 00:00
authors: ["Wang, L."]
year: 2024
doi: "10.3389/fonc.2024.1281922"
citekey: wangMammographyDeepLearning2024
status: draft
aliases:
- "Wang, L. (2024)"
---

> [!abstract]
> This paper provides an extensive overview of medical imaging techniques and deep learning approaches applied to breast cancer detection. It surveys preprocessing methods, model architectures, performance metrics, and highlights the potential of convolutional neural networks (CNNs), transfer learning, and attention-based models in improving diagnostic accuracy for mammography.

## Summary

* Comprehensive review of medical imaging and deep learning (DL) applications for [Breast Cancer Detection](../permanent/breast-cancer-detection.md).
* Evaluates performance across public datasets (DDSM, CBIS-DDSM, INbreast, MIAS, BreaKHis).

## Notes

### Medical Imaging Techniques

* [Mammography](../../../../permanent/mammography.md)
    * Utilises low-dose X-rays to generate breast images that radiologists use to find lumps, calcifications and distortions.
    * Recommended for women over 40.
    * Limitations:
        * reduced sensitivity for women with dense breast tissue.
* [Digital Breast Tomosynthesis (DBT)](../../../../permanent/digital-breast-tomosynthesis-dbt.md) 
    * Uses x-rays to generate 3d breast images, useful for detect BC in dense breasts.
    * Faces inter-observer variability which can affect accuracy.
* [Ultrasound Imaging](../../../../permanent/ultrasound-imaging.md)
    * High-frequency sound waves to produce detailed images of breast abnormalisities, especially in women with dense breasts.
    * Does not involved radiation, so it's generatlly safer.
* [Magnetic Resonance Imaging (MRI)](../../../../permanent/magnetic-resonance-imaging-mri.md)
    * Uses magnetic fields and radio waves to image soft tissue without ionizing radiotaion.
    * Effective for dense breasts and high-risk patients.
    * Tends to have lower [Specificity](Specificity.md) (more false positives)
* [Positron Emission Tomography (PET)](../../../../permanent/positron-emission-tomography-pet.md)
    * Detects gamma rays emitted by radiotracers to map metabolic activity and identify malignant lesions.
    * Creates 3d images of internal body.

### Deep learning techniques

* DL architectures covered:
    * [CNN](../../permanent/convolutional-neural-network.md)
    * [Transfer Learning](../../permanent/transfer-learning.md)
    * Ensemble Learning
        * [Stacking](../../../../permanent/stacking.md)
        * [Boosting](../../../../permanent/boosting.md)
        * [Bagging](../../../../permanent/bagging.md)
    * [Attention](../../permanent/attention-mechanism.md) based models ([Transformer](../../permanent/transformer.md))
        * SE-Net
        * Channel Attention Networks (CAN)

#### Preprocessing

* Denoising techniques:
    * [Median filter](Median%20filter.md)
    * [Wiener Filter](Wiener%20Filter.md)
    * [Non-local means filter](Non-local%20means%20filter.md)
    * [Total variation (TV) denoising](Total%20variation%20(TV)%20denoising.md)
    * [Wavelet-based denoising](Wavelet-based%20denoising.md)
    * [Gaussian filter](Gaussian%20filter.md)
    * [Anisotropic Diffusion](Anisotropic%20Diffusion.md)
    * [BM3D Denoising](BM3D%20Denoising.md)
    * [Auto-Encoder](../../../../permanent/autoencoder.md)
* Normalisation methods typically applied after denoising, like [min-max normalisation](min-max%20normalisation.md)

#### Datasets

* [DDSM](https://marathon.csee.usf.edu/Mammography/database.html) – Original digitised mammography dataset.  
* [CBIS-DDSM](../reference/a-curated-mammography-data-set-for-use-in-computer-aided-detection-and-diagnosis-research.md) – Curated subset with improved annotations.  
* INbreast, MIAS, and BreaKHis – Alternative datasets with varied imaging quality and pathology balance.  
* Data augmentation (rotations, flips, noise injection) helps overcome limited sample sizes.

#### Performance Metrics

* [Accuracy](Accuracy.md)
    * measures the proportion of correct predictions made by the model.
    * $\frac{TP + TN}{TP+TN+FP+FN}$
* [Precision](Precision.md)
    * Measures the proportion of true positive predictions out of all positive predictions made by the model.
      * $\frac{TP}{TP + FP}$
* [Sensitivity](Sensitivity.md)
    * Measures the proportion of true positive predictions out of all actual positive cases in the dataset.
    * $\frac{TP}{TP + FN}$
* [F1-Score](../../../../permanent/f1-score.md)
    * A composite metric that balances [Precision](Precision.md) and [Sensitivity](Sensitivity.md).
    * $2 \times \frac{Precision \times Recall}{Precision + Recall}$
* [Area Under the Curve (AUC)](Area%20Under%20the%20Curve%20(AUC))
    * Evaluates the model’s ability to distinguish between positive and negative cases across various threshold values.
    * A higher AUC indicates better discriminatory performance.
* [Mean-Squared Error](../../../../permanent/mean-squared-error.md)
    * Measures the average squared difference between predicted and actual values, used primarily for regression tasks.
    * $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

## Key Findings

* Deep learning significantly improves sensitivity and specificity compared to conventional CAD systems.  
* Transfer learning and ensembles increase performance across heterogeneous datasets.  
* Attention-based and Transformer architectures further enhance lesion localisation and reduce false positives.  
* Model interpretability is essential for radiologist trust and clinical adoption.

## Limitations and Future Directions

* Performance still constrained by dataset bias and limited public data availability.  
* Need for cross-institutional generalisation and federated learning approaches.  
* Integration of explainability, uncertainty estimation, and radiologist feedback loops recommended for deployment.

## Related

* [Lee et al., 2017](a-curated-mammography-data-set-for-use-in-computer-aided-detection-and-diagnosis-research.md) – describes the CBIS-DDSM dataset used in multiple models reviewed.  
* [Chollet, 2018](https://www.manning.com/books/deep-learning-with-python) – reference for CNN design and training workflow.

## Reference

* Wang, L. (2024). Mammography with deep learning for breast cancer detection. Frontiers in Oncology, 14, 1281922. https://doi.org/10.3389/fonc.2024.1281922
