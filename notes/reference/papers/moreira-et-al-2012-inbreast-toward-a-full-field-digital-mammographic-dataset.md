---
title: "Moreira et al. (2012) - INbreast: Toward a Full-field Digital Mammographic Database"
year: 2012
doi: "10.1016/j.acra.2011.09.014"
citekey: "10.1016/j.acra.2011.09.014"
date: 2025-10-16 00:00
modified: 2025-10-16 00:00
status: draft
tags:
- BreastCancerDetection
- DeepLearning
- Dataset
- Mammography
---

## Summary

 Introduces [InBreast](../../permanent/inbreast.md) a dataset designed for the development and evalation of computer-aided detection (CADe) and computer-aided diagnosis (CADx) algorithm.

Acquired at the Breast Centre Hospitalar de São João (CHSJ), Porto, it contains precisely annotated digital Mammograms with XML-based contours of lesions and associated metadata such as BI-RADS categories, breast density, patient age, and pathology results.

Device used was MammoNovation Siemens full-field digital mammography (as opposed to digitised film images) in the datasets prior like [CBIS-DDSM Mammography Dataset](../../permanent/cbis-ddsm-mammography-dataset.md), with a solid-state detector of amorphous selenium was used.

### Dataset Details

* Developed by the **University of Porto / INESC Porto** under hospital and ethics committee approval.  
* Acquisition system: **Siemens MammoNovation FFDM** (amorphous selenium detector).  
* Image specs:
    * Pixel size: 70 μm.
    * Bit depth: 14-bit grayscale.
    * Resolution: 2560 × 3328 or 3328 × 4084 (depending on compression plate).
    * Format: [DICOM](../../permanent/dicom.md).
* Dataset size:
    * **115 cases**, 410 images.
    * 90 cases with both breasts (4 images per case: CC + MLO).
    * 25 single-breast (mastectomy) cases (2 images per case).
* Contains:
    * **Normal**, **benign**, and **malignant** images.
    * Multiple lesion types: **masses**, **calcifications**, **architectural distortions**, **asymmetries**, and **spiculated regions**.
* Annotations:
    * Detailed **XML contours** for all regions of interest.
    * Labels for lesion type and location.
    * **Seven annotation types:** asymmetry, calcification, cluster (of microcalcifications), mass, distortion, spiculated region, pectoral muscle.
    * Annotation tool: **OsiriX (Mac)** PACS workstation.
    * Dual-specialist validation; disagreements resolved by consensus.
* Metadata:
    * **BI-RADS categories (0–6)**.
    * **ACR breast density** classification (1–4).
    * **Patient age**, **family history**, and **pathology** results.
    * Biopsy results included for BI-RADS 3–6.
* Distribution:
    * Hosted at [INESC Porto](http://medicalresearch.inescporto.pt/breastresearch/GetINbreastDatabase.html).
        * Which is a broken link. Found it on Google drive: https://drive.google.com/file/d/19n-p9p9C0eCQA1ybm6wkMo-bbeccT_62/view
    * Suggested **train/test split** included.
    * Available for **public research use** with ethics compliance.

### Technical Validation

* 56 biopsy-confirmed cases: 45 malignant, 11 benign.  
* Strong inter-radiologist consensus ensured annotation precision.  
* The high-resolution contours enable **pixel-level segmentation** evaluation—superior to DDSM’s coarse ROIs.  
* Provides distributions by BI-RADS class, lesion type, breast density, and age.  
* Demonstrates natural clinical diversity (including difficult cases and post-surgery exams), increasing model robustness potential.

### Findings Breakdown

* **Lesion distribution:**
  * 116 masses across 107 images (~1.1 per image).
  * 6,880 individual calcifications annotated.
  * 27 clusters of microcalcifications (MCCs).
  * Prominent representation of calcifications (reflecting clinical prevalence).  
* **Lesion size:** average 479 mm² (SD = 619 mm²; range = 15–3689 mm²).  
* **Age distribution:** broad, with representative real-world variability.  
* **Density distribution:** includes ACR categories 1–4.  

### Significance

* First **public FFDM dataset** available to researchers (as of 2012).  
* Designed to **overcome DDSM’s limitations** in precision, compression, and clinical completeness.  
* Enables:
  * Benchmarking of **deep learning models** for segmentation and classification.
  * **Shape-based malignancy analysis** and microcalcification clustering studies.
  * Evaluation of **CAD systems** under realistic imaging conditions.  
* Provides an important bridge from digitized to **natively digital mammography research**.

### Related

* [CBIS-DDSM Mammography Dataset](../../permanent/cbis-ddsm-mammography-dataset.md) — CBIS-DDSM dataset, modernized from DDSM with improved segmentations and standardized splits.  
* [Wang, L. (2024)](../../../../reference/papers/3070/mammography-with-deep-learning-for-breast-cancer-detection.md) — review of deep learning methods using mammography datasets.

### Reference

* Moreira, I. C., Amaral, I., Domingues, I., Cardoso, A., Cardoso, M. J., & Cardoso, J. S. (2012). **INbreast: Toward a full-field digital mammographic database.** *Academic Radiology, 19*(2), 236–248. https://doi.org/10.1016/j.acra.2011.09.014
