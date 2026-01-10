---
title: Mammography
date: 2025-10-15 00:00
modified: 2025-10-15 00:00
summary: "A breast cancer screening method"
tags:
- MedicalImaging
- BreastCancerDetection
cover: /_media/mammography-cover.png
---

As part of my final project for my BSc, I worked on Breast Cancer Detection. This note was made while doing background research for that topic. See topic #BreastCancerDetection

X-ray **Mammography** is a breast cancer screening method and remains one of the more effective population-wide tools for early detection of Breast Cancer [^1]. The breast is compressed between two plates to spread the tissue and reduce motion blur evenly. A typical screening exam records two views of the breast: [Caniocaudal (CC)](caniocaudal-cc.md), a top-to-bottom view, and [Mediolateral (MLO)](mediolateral-mlo.md), a side view.

![mammography-screen-views.png](../_media/mammography-screen-views.png)

When reading a mammogram, a radiologist looks for specific abnormalities: masses, calcifications, distortion of breast tissue, or asymmetries when comparing the two breasts and two views.

To standardise reporting, radiologists use the [Breast Imaging Reporting and Data System (BI-RADS)](breast-imaging-reporting-and-data-system-bi-rads.md), which assigns a category from 0 to 6 indicating the level of suspicion:

* **0** - incomplete: additional imaging needed.
* **1** - negative.
* **2** - benign.
* **3** - probably benign: short-interval follow-up suggested.
* **4** - suspicious: biopsy should be considered.
* **5** - highly suggestive of malignancy.
* **6** - known biopsy with proven malignancy.

Categories 4 and 5 typically warrant a biopsy to confirm or rule out cancer.

![mammography-bi-rads.png](../_media/mammography-bi-rads.png)

Radiologists also classify breast composition by density using four categories:

* **A** - almost entirely fatty
* **B** - scattered fibroglandular densities
* **C** - heterogeneously dense
* **D** - extremely dense

Higher breast density both increases breast cancer risk and reduces mammographic sensitivity, as dense tissue appears white on mammograms (the same appearance as potential tumours), effectively masking lesions.

![mammography-breast-density-categories.png](../_media/mammography-breast-density-categories.png)

Modern screening often utilise [Digital Breast Tomosynthesis (DBT)](digital-breast-tomosynthesisd-dbt.md), or "3D mammography." Unlike standard 2D mammography, DBT captures multiple X-ray projections from different angles to reconstruct the breast in "slices." This minimises the effect of overlapping tissue, improving detection rates in dense breasts.

On of the canonical Mammography datasets is [CBIS-DDSM Mammography Dataset](cbis-ddsm-mammography-dataset.md).

## References

[^1]: Misra, S., Solomon, N. L., Moffat, F. L., & Koniaris, L. G. (2010). *Screening criteria for breast cancer*. **Advances in Surgery, 44**, 87–100. [https://doi.org/10.1016/j.yasu.2010.05.008](https://doi.org/10.1016/j.yasu.2010.05.008)