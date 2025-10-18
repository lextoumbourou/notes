---
title: "RSNA Breast Cancer Detection Competition"
date: 2025-10-18 00:00
modified: 2025-10-18 00:00
status: draft
tags:
- BreastCancerDetection
- Mammography
- DeepLearning
- RSNA
- Kaggle
aliases:
- "RSNA 2023 Breast Cancer Competition"
---

The RSNA Breast Cancer Detection challenge was a 2023 Kaggle competition hosted by Radiological Society of North America (RSNA) to identify malignant cases in screening [Mammograms](../../../../permanent/mammogram.md).

Participants develop models using DICOM-format breast images and metadata to predict the likelihood of cancer while minimizing false positives.

To develop models that **detect breast cancer from screening mammograms**.  
A successful model assists radiologists by flagging probable cancer cases while reducing false positives. Improved automation can enhance diagnostic efficiency, reduce unnecessary biopsies, and expand access to screening globally.

## Dataset Overview

They develop a new mammography dataset that contains images for 11913 patients, with about 4 images per patient (two per breast, two views)

Like [CBIS-DDSM Mammography Dataset](../../../../permanent/cbis-ddsm.md) it also contains metadata files:
* `train.csv` and `test.csv` — metadata describing each image.  
* `site_id`: hospital source  
* `patient_id`: unique identifier  
* `image_id`: unique DICOM identifier  
* `laterality`: left/right breast  
* `view`: orientation (usually two per breast)  
* `age`: patient age  
* `implant`: breast implant presence  
* `density`: breast tissue density (A–D)  
* `machine_id`: imaging device  
* **Training only**:
* `cancer`: binary target label (1 = malignant)  
* `biopsy`: whether follow-up biopsy performed  
* `invasive`: cancer invasiveness flag  
* `BIRADS`: radiologist score (0 = follow-up required, 1 = negative, 2 = normal)  
* `difficult_negative_case`: unusually hard negatives  

* **Sample submission**:  
`sample_submission.csv` — shows correct format for predictions using `prediction_id,cancer`.

## Evaluation Metric

**Probabilistic F1 score (pF1)** - an extension of F1 that uses predicted probabilities instead of binary outputs.  

Higher scores indicate better balance between sensitivity and precision over probabilistic predictions.

## Solution Summaries


