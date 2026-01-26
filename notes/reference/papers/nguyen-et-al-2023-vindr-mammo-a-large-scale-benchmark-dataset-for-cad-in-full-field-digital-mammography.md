---
title: "Nguyen et al. (2023) - VinDr-Mammo: A Large-Scale Benchmark Dataset for CAD in Full-Field Digital Mammography"
year: 2023
doi: "10.1038/s41597-023-02100-7"
citekey: "nguyen2023vindrmammo"
date: 2025-10-20 00:00
modified: 2025-10-20 00:00
status: draft
tags:
- BreastCancerDetection
- DeepLearning
- Dataset
- Mammography
---

## Summary

[VinDr-Mammo](../../permanent/vindr-mammo.md) is a large-scale Vietnamese full-field digital [Mammography](../../permanent/mammography.md) (FFDM), which consists of 5,000 exams (20,000 images) created to support [Computer-Aided Detection (CADe)](../../../../permanent/computer-aided-detection-cade.md) and [Computer-Aided Diagnosis (CADx)](../../../../permanent/computer-aided-diagnosis.md) research.

Contains 4 standard views for each patient (L/R CC and MLO) and provides:

- Breast-level BI-RADS assessment.
- Breast density (ACR categories A-D)
- Extensive lesion-level bounding-box annotations for masses, calcifications, asymmetries, distortions and other suspicious features.
- Double-reading with abritration. That is, two radiologists read each exam, if they disagree, a third one resolve the conflict.

One of the largest, publically accessible FFDM datasets.

## Dataset

* Time window: 2018–2020  
* Equipment vendors: Siemens, IMS, Planmed  
* All images are for-presentation DICOMs (no raw for-processing images saved by hospitals).  
* Images de-identified:  
    - DICOM PHI scrubbed  
    - Text burned into corners masked out 
* Source:
    * Images sourced from two major hospitals in Hanoi, Vietnam:
        - Hospital 108 (H108)
        - Hanoi Medical University Hospital (HMUH)
- Format: DICOM
- Views: CC & MLO (left & right)
- Total: 20,000 images / 5,000 exams
- Stored in per-study folders with encoded IDs.

### Labels and Annotations

* Breast-level (per image)
    * [BI-RADS](../../permanent/breast-imaging-reporting-and-data-system-bi-rads.md) scale:
        * 1, 2, 3, 4, 5 (no 6; pathology unavailable)
    * Breast density: A-D
    * Metadata also has: laterality, view, image size, split.
* Lesion level:
    * Bounding box coordinates
    * Categories:
        * mass
        * suspicious calcification
        * asymmetry (global / focal)
        * architectual distortion
        * associated findings: skin thickening, skin retraction, nipple retraction, suspicious lymph node
    * Each lession includes own BI-RADS assessment.
* Annotation protocol:
    * Double-reading by experienced radiologists.
    * Arbitraty third read if disagreement.
    * Readers average 19 years clinical experience and interpret ~10k-15k mammograms/year
    * Web-based tool: VinDR Lab (OHIF-based viewer)
* Stratified Data Split
    * 4,000 training exams
    * 1,000 test exams
    * Iterative stratification preserving distributions across:
          - BI-RADS  
          - density  
          - lesion categories  
## Technical Validation

* Privacy & Data Quality
    * DICOM metadata reviewed manually post-scrubbing  
    * Image inspection to ensure absence of PHI  
    * Annotation errors prevented by built-in QA rules (e.g., cannot mark a lesion while selecting BI-RADS 1)  
* Dataset Characteristics (Selected)
    * BI-RADS distribution (overall 10,000 breasts):
        - BI-RADS 1: 67.0%  
        - BI-RADS 2: 23.4%  
        - BI-RADS 3: 4.7%  
        - BI-RADS 4: 3.8%  
        - BI-RADS 5: 1.1%  
* Density distribution:
    - A: 0.5%  
    - B: 9.5%  
    - C: 76.5% (dominant)  
    - D: 13.5%  
* Lesion counts (entire dataset):
    - Masses: **1,226**  
    - Suspicious calcifications: **543**  
    - Asymmetries (global, focal): **392**  
    - Architectural distortions: **119**  
    - Lymph nodes / retractions: <60 each  
## Significance

VinDr-Mammo addresses several key gaps in mammography AI research:
1. **Scale & diversity:**  
    * Largest *public* FFDM dataset with rich annotation detail.
2. **Population representation:**  
    * Introduces Southeast Asian data—critical given known domain shift across ethnicities, vendors, and clinical pipelines.
3. **Realistic screening distribution:**  
    * Mix of diagnostic + screening exams, and clinically realistic prevalence.
4. **Bounding-box–level lesion annotations:**  
    * Suitable for CADe, object detection, and weakly supervised methods.
5. **Standardized split** ensures reproducibility across publications.

Limitations:  
- No pathology-proven labels → BI-RADS 4/5 are only *suspicious*, not confirmed malignancies.  
- Some findings have very low sample counts (<40).  
- DICOMs are not fully compliant with all processing libraries.  

## Comparisons

[Moreira et al. (2012) - INbreast: Toward a Full-field Digital Mammographic Database](moreira-et-al-2012-inbreast-toward-a-full-field-digital-mammographic-dataset.md)

| Feature              | INbreast                  | VinDr-Mammo                                 |
| -------------------- | ------------------------- | ------------------------------------------- |
| Size                 | **115 cases**             | **5,000 exams** — *~40× larger*             |
| Population           | Portuguese                | Vietnamese (underrepresented population)    |
| Annotation precision | Pixel-accurate contours   | Bounding boxes (faster + scalable)          |
| Pathology labels     | Included for many lesions | Not available                               |
| Imaging              | Siemens FFDM              | Siemens / IMS / Planmed                     |
| Use cases            | CADx, segmentation        | CADe, detection, multi-label classification |

**Key distinction:**  
INbreast remains the gold standard for **pixel-level precision**, but is tiny; VinDr-Mammo is designed for **scale**, population diversity, and bounding-box detection workflows.  

[CBIS-DDSM Mammography Dataset](../../permanent/cbis-ddsm-mammography-dataset.md)

| Feature | CBIS-DDSM | VinDr-Mammo |
|--------|-----------|-------------|
| Imaging | Digitised film | Full-field digital mammography |
| Size | ~1,644 cases | 5,000 exams |
| Lesion types | Mass, calcification | Mass, calcification, distortion, multiple asymmetries, associated features |
| Annotations | Updated segmentations | Bounding boxes |
| Pathology labels | Yes | No |
| Image quality | Variable (film artefacts, scanner noise) | High-quality clinical FFDM |

CBIS-DDSM is rich and pathology-linked but suffers from film-scanner limitations.  
VinDr-Mammo provides modern FFDM, higher consistency, vastly larger scale, and more comprehensive lesion categories.  

## Reference

* Nguyen, H. T., Nguyen, H. Q., Pham, H. H., et al. (2023).   **VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography.** *Scientific Data, 10*(1), 277. https://doi.org/10.1038/s41597-023-02100-7  