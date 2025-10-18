---
title: A curated mammography data set for use in computer-aided detection and diagnosis research
year: 2017
doi: "10.1038/sdata.2017.177"
citekey: "10.1038/sdata.2017.177"
date: 2025-10-16 00:00
modified: 2025-10-16 00:00
status: draft
tags:
- BreastCancerDetection
- DeepLearning
- Dataset
- Mammography
aliases:
- "Lee et al., 2017"
---

> [!abstract]
> The CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated, standardized mammography dataset derived from the original Digital Database for Screening Mammography (DDSM). It provides decompressed DICOM images, curated metadata, improved lesion segmentations, and standardized training/test splits to enable reproducible research in computer-aided detection (CADe) and diagnosis (CADx) of breast cancer.

## Summary

* Introduces [CBIS-DDSM Mammography Dataset](../../../../permanent/cbis-ddsm.md). 

### Background

- [Computer-Aided Diagnosis (CADx)](../../../../permanent/computer-aided-diagnosis.md) (diagnosis) / [Computer-Aided Detection (CADe)](../../../../permanent/computer-aided-detection-cade.md) (detection) mammography research hindered by lack of consistent datasets.
- Existing datasets fragmented, compressed in obsolete formats, and had imprecise ROI annotations.
- **CBIS-DDSM** resolves these issues, offering decompressed images, improving segmentation, curated metadata, and standardised train/test splits.

### Dataset Details

* An updated and standardised version of Digital Database for Screening Mammography.
* Contains:
    * 753 calcification cases.
    * 891 mass cases.
    * Benign and malignant classifications with ground truth labels conformed by experts.
* Format: 16-bit grayscale [DICOM](DICOM.md) formats (from decompressed LJPEG).
* Annotations:
    * updated ROI masks
    * bounding boxes
    * lesion segmentations using a local level set algorithm (modified [Chan-Vese](../../../../permanent/Chan-Vese.md)).
* Metadata:
    * BI-RADS descriptors
    * Breast Density
    * Abnormality type
    * Pathology
    * Subtlety rating consolidated in CSVs
    * Train/test split: 80/20 stratified by BI-RADS cateogry to ensure balanced difficulty.

### Methods

* They converted the legacy LJPEG to DICOM using modified PVRG-JPEG codec.
* Parsed the `.ics` and `.OVERLAY` files into unified CSV.
* Removed 339 questionable mass images after review by Mammographer.
* Automated segmentation algorithm validated against hand-drawn radiologist ROIs (Dice coefficient 0.79 ± 0.11 vs. 0.40 ± 0.20 for original DDSM).
* Processing code (Python 2.7.9) Github:
    * https://github.com/fjeg/ddsm_tools
* Data hosted on TCIA: 
    * https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

### Technical Validation

* Significant improvement in segmentation algorithm: $\text{Wilcoxon }p < 5.5×10^{-19}$
* Validation performed on 118 manually annotated images across four BI-RADS density categories.

## Related

* [Wang, L. (2024)](mammography-with-deep-learning-for-breast-cancer-detection.md) - literature review of Deep Learning models applied to Breast Cancer Detection.
* Original [DDSM](https://marathon.csee.usf.edu/Mammography/database.html) - 2,620 film mammography cases (scanned), basis for CBIS-DDSM.

## Reference

* Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. Scientific Data, 4(1), 170177. https://doi.org/10.1038/sdata.2017.177