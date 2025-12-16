---
title: CBIS-DDSM Mammography Dataset
date: 2025-10-18 00:00
modified: 2025-10-18 00:00
status: draft
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

See [Lee et al. (2017) - A curated mammography data set for use in computer-aided detection and diagnosis research](../../../reference/papers/3070/lee-et-al-2017-a-curated-mammography-data-set-for-use-in-computer-aided-detection-and-diagnosis-research.md).

**CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** is a [Mammography](mammography.md) dataset, derived from the original Digital Database for Screening Mammography (DDSM). The idea of CBIS-DDSM was to provide a standardise Mammography dataset towards an [ImageNet](../../../permanent/ImageNet.md) for Mammography.

In this article, I'm going to walk through how the dataset works, and share some preprocessing examples.

The dataset can be download from from https://www.cancerimagingarchive.net/collection/cbis-ddsm.

The image dataset 164GB compressed dataset which uncompresses to around 180GB.

```python
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
```

```python
DATASET_ROOT = Path("/Users/lex/datasets/CBIS-DDSM")

!cd {DATASET_ROOT} && du -sh *
```


## Metadata File Review

There is 2 files provided for each split, representing either calcification or mass abnormalities found in the breast.

- `calc_case_description_${train|test}_set.csv`
- `mass_case_description_${train|test}_set.csv`

Here we load each file, then concat together to give us one dataset file per split.
```python
train_mass_df = pd.read_csv(DATASET_ROOT / "mass_case_description_train_set.csv")
train_calc_df = pd.read_csv(DATASET_ROOT / "calc_case_description_train_set.csv")
train_df = pd.concat([train_mass_df, train_calc_df])
train_mass_df = train_calc_df = None
train_df.head(1)
```

```python
test_mass_df = pd.read_csv(DATASET_ROOT / "mass_case_description_test_set.csv")
test_calc_df = pd.read_csv(DATASET_ROOT / "calc_case_description_test_set.csv")
test_df = pd.concat([test_mass_df, test_calc_df])
test_mass_df = test_mass_df = None
test_df.head(1)
```

```python
all_df = pd.concat([train_df, test_df])
```

```python
metadata_df = pd.read_csv(DATASET_ROOT / "metadata.csv")
metadata_df.head(1)
```

CBIS-DDSM is a subset of the original DDSM dataset which contained 2,2620 scanned mammography studies, it has 1,566 studies.

```python
images_per_patient = all_df.groupby("patient_id").size()
print(f"Total patients: {len(images_per_patient)}")
```

They separate the studies by the abnormality type present, either "mass" or "calcification".


The paper claims that there's 891 mass cases, although the actual dataset appears to have 892 mass abnormalities.

```python
len(all_df[all_df["abnormality type"] == "mass"].patient_id.unique())
```

The paper also describes 753 calcification abnormalities, which matches what we see in the paper.

```python
len(all_df[all_df["abnormality type"] == "calcification"].patient_id.unique())
```

We know that a Mammogram consists of 2 images per breast, a top-down view (MLO) or a side-view CLO.

Looking at the distribution of images per patient, we have about 1005 complete mammograms, with a number of single image 

```python
fig, ax = plt.subplots(figsize=(10, 5))
images_per_patient.value_counts().sort_index().plot(kind='bar', ax=ax, color='#3498db', edgecolor='black')
ax.set_title('Distribution of Images per Patient', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Images')
ax.set_ylabel('Number of Patients')
for i, v in enumerate(images_per_patient.value_counts().sort_index().values):
    ax.text(i, v + 2, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()
```
## Image Extraction

The originally DDSM images where distributed in an obselete JPEG form called LJPEG. Which apparently the only library capable of decompressing with "last updated in 1993".

They modified the original codec, and then extract the raw pixels, converting them into 16-bit DICOM format, which is the standard for medical images. The extraction is entirely lossless, although the fact that the original images are scans and not some more sophistocated like Full-Field Digital Mammography (as in the case of other datasets like InBreast), will limit the capability of DDSM somewhat.









