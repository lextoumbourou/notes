---
category: note
title: CBIS-DDSM Mammography Dataset
date: 2025-10-18 00:00
modified: 2026-09-13 13:08
summary: ImageNet for Mammography
cover: /_media/cbis-ddsm-patient-assets.png
tags:
- BreastCancerDetection
- MedicalImaging
- Mammography
notebook:
  markdownLinks: true
  python: ../../.venv/bin/python
---

**CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** [^1] is a [Mammography](mammography.md) dataset for [Computer-Aided Detection (CADe)](computer-aided-detection-cade.md) and [Computer-Aided Diagnosis (CADx)](computer-aided-diagnosis.md), derived from an earlier dataset, the Digital Database for Screening Mammography (DDSM) [^2]. This note comes from background research for my Breast Cancer Detection BSc final project.

The idea of CBIS-DDSM was to provide a standardised mammography dataset towards an [ImageNet](ImageNet.md) for mammography. Though a few mammography datasets already existed: the DDSM itself [^2], the Mammographic Imaging Analysis Society (MIAS) database [^3], and the Image Retrieval in Medical Applications (IRMA) project [^4], they were limited by accessibility and data quality issues.

DDSM was already a promising dataset for this purpose, comprising 2620 scanned mammography studies from multiple hospitals. It contains ROI annotations and [Breast Imaging Reporting and Data System (BI-RADS)](breast-imaging-reporting-and-data-system-bi-rads.md) labels for a series of Mammography studies, along with extensive metadata. However, it had several problems: inaccurate region-of-interest annotations, personal health information in some images, and an obsolete file format (LJPEG).

The authors of the CBIS-DDSM subset stripped the dubious annotations and the examples containing PII. They also wrote a conversion tool for LJPEG and converted the images into TIFF files, which are then stored as [DICOM](dicom.md) files, the standard for medical images, to create CBIS-DDSM.

They also included convenience images, including the region-of-interest mask and the cropped region. You can see an example of it later in the article.

Finally, they improved the accuracy of the existing region-of-interest annotations by applying the [Chan-Vese Algorithm](chan-vese-algorithm.md), initialised with the original contours, but only to the mass examples, not the calcifications. In the figure below, in red, the original annotations; in blue, some example annotations created by physicians; and in green, the annotations derived from the Chan-Vese model, which clearly improve on the original annotations.

![Figure 2 from Lee et al demonstrating the Chan-Vese algorithm for improving ROI annotations](../_media/cbis-ddsm-figure-2.png)

I'm going to walk through how the dataset works in this rendered notebook.

The dataset can be downloaded from the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cbis-ddsm).

I downloaded the **Images** dataset to the `~/datasets/CBIS-DDSM` folder, which uncompresses into the `./CBIS-DDSM` directory.

The image dataset is a 164GB compressed dataset, which uncompresses to around 180GB.

```python
from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd
import pydicom

pd.set_option('display.max_colwidth', None)
```
<!-- nb-output hash="510065e1bcee884c" format="html" -->

<!-- /nb-output -->

```python
import subprocess

DATASET_ROOT = Path("/Users/lex/datasets/CBIS-DDSM")

csv_files = sorted(path.name for path in DATASET_ROOT.glob("*.csv"))
print(subprocess.check_output(
    ["du", "-sh", *csv_files, "CBIS-DDSM"],
    cwd=DATASET_ROOT,
    text=True,
), end="")
```
<!-- nb-output hash="c1fcedc209e8efef" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">512K	calc_case_description_test_set.csv
1.0M	calc_case_description_train_set.csv
512K	mass_case_description_test_set.csv
1.0M	mass_case_description_train_set.csv
3.0M	metadata.csv
180G	CBIS-DDSM
</pre>
</div>
<!-- /nb-output -->

2 files are provided for each split, representing either **calcification** or **mass abnormalities** found in the breast.

- `calc_case_description_{train|test}_set.csv`
- `mass_case_description_{train|test}_set.csv`

```python
train_mass_df = pd.read_csv(DATASET_ROOT / "mass_case_description_train_set.csv")
train_mass_df.head(1).T
```
<!-- nb-output hash="2f7ac30064e48d4a" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><div>
<style>.dataframe tbody tr th:only-of-type { vertical-align: middle; } .dataframe tbody tr th { vertical-align: top; } .dataframe thead th { text-align: right; }</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>patient_id</th>
      <td>P_00001</td>
    </tr>
    <tr>
      <th>breast_density</th>
      <td>3</td>
    </tr>
    <tr>
      <th>left or right breast</th>
      <td>LEFT</td>
    </tr>
    <tr>
      <th>image view</th>
      <td>CC</td>
    </tr>
    <tr>
      <th>abnormality id</th>
      <td>1</td>
    </tr>
    <tr>
      <th>abnormality type</th>
      <td>mass</td>
    </tr>
    <tr>
      <th>mass shape</th>
      <td>IRREGULAR-ARCHITECTURAL_DISTORTION</td>
    </tr>
    <tr>
      <th>mass margins</th>
      <td>SPICULATED</td>
    </tr>
    <tr>
      <th>assessment</th>
      <td>4</td>
    </tr>
    <tr>
      <th>pathology</th>
      <td>MALIGNANT</td>
    </tr>
    <tr>
      <th>subtlety</th>
      <td>4</td>
    </tr>
    <tr>
      <th>image file path</th>
      <td>Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/000000.dcm</td>
    </tr>
    <tr>
      <th>cropped image file path</th>
      <td>Mass-Training_P_00001_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.108268213011361124203859148071588939106/1.3.6.1.4.1.9590.100.1.2.296736403313792599626368780122205399650/000000.dcm</td>
    </tr>
    <tr>
      <th>ROI mask file path</th>
      <td>Mass-Training_P_00001_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.108268213011361124203859148071588939106/1.3.6.1.4.1.9590.100.1.2.296736403313792599626368780122205399650/000001.dcm\n</td>
    </tr>
  </tbody>
</table>
</div></div>
</div>
<!-- /nb-output -->

Before we get to the metadata, let's take a look at some major bugs with the provided CSV.

### Addressing Inconsistent Image Mappings {id="addressing inconsistent image mappings"}

As you can see, each row in the CSV files contains references to 3 DICOM files:

- `image file path` - the full mammogram
- `ROI mask file path` - binary mask of the region of interest
- `cropped image file path` - cropped region containing the abnormality

However, the DICOM filenames don't match the files downloaded from the **Images** dataset in a quite confusing way.

The CSVs reference files like `000000.dcm` or `000001.dcm`, but the actual files are named `1-1.dcm` or `1-2.dcm`. Even worse, the mapping between these naming conventions is inconsistent. Additionally, some entries in the `ROI mask file path` column incorrectly point to cropped images rather than actual binary masks.

Thankfully, Andrés Sarmiento created a [tool](https://gitlab.com/ACSG-64/cbis-ddsm-description-correction-and-verification-tool) that fixes these issues by interrogating the mask files to determine if they're crops or masks, and correcting the filepath references. The corrected CSV files are available as a [HuggingFace dataset](https://huggingface.co/datasets/ACSG-64/CBIS-DDSM-description-corrected).

I downloaded the correct CSV to `~/datasets/CBIS-DDSM/fixed-csv`, and the rest of the notebook will use it accordingly.

## Train / Test Data {id="train / test data"}

As mentioned, the training and test data are split by the abnormality type present in the scan: calcification or mass. I find it easiest to combine the training into a single file:

```python
train_mass_df = pd.read_csv(DATASET_ROOT / "fixed-csv" / "mass_case_description_train_set.csv")
train_calc_df = pd.read_csv(DATASET_ROOT / "fixed-csv" / "calc_case_description_train_set.csv")
train_df = pd.concat([train_mass_df, train_calc_df])
train_mass_df = train_calc_df = None
train_df.head(1).T
```
<!-- nb-output hash="651cf8ec50c19681" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><div>
<style>.dataframe tbody tr th:only-of-type { vertical-align: middle; } .dataframe tbody tr th { vertical-align: top; } .dataframe thead th { text-align: right; }</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>patient_id</th>
      <td>P_00001</td>
    </tr>
    <tr>
      <th>breast_density</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>left or right breast</th>
      <td>LEFT</td>
    </tr>
    <tr>
      <th>image view</th>
      <td>CC</td>
    </tr>
    <tr>
      <th>abnormality id</th>
      <td>1</td>
    </tr>
    <tr>
      <th>abnormality type</th>
      <td>mass</td>
    </tr>
    <tr>
      <th>mass shape</th>
      <td>IRREGULAR-ARCHITECTURAL_DISTORTION</td>
    </tr>
    <tr>
      <th>mass margins</th>
      <td>SPICULATED</td>
    </tr>
    <tr>
      <th>assessment</th>
      <td>4</td>
    </tr>
    <tr>
      <th>pathology</th>
      <td>MALIGNANT</td>
    </tr>
    <tr>
      <th>subtlety</th>
      <td>4</td>
    </tr>
    <tr>
      <th>image file path</th>
      <td>Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/1-1.dcm</td>
    </tr>
    <tr>
      <th>cropped image file path</th>
      <td>Mass-Training_P_00001_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.108268213011361124203859148071588939106/1.3.6.1.4.1.9590.100.1.2.296736403313792599626368780122205399650/1-2.dcm</td>
    </tr>
    <tr>
      <th>ROI mask file path</th>
      <td>Mass-Training_P_00001_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.108268213011361124203859148071588939106/1.3.6.1.4.1.9590.100.1.2.296736403313792599626368780122205399650/1-1.dcm</td>
    </tr>
    <tr>
      <th>breast density</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>calc type</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>calc distribution</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div></div>
</div>
<!-- /nb-output -->

We do the same for the test set:

```python
test_mass_df = pd.read_csv(DATASET_ROOT / "fixed-csv" / "mass_case_description_test_set.csv")
test_calc_df = pd.read_csv(DATASET_ROOT / "fixed-csv" / "calc_case_description_test_set.csv")
test_df = pd.concat([test_mass_df, test_calc_df])
test_mass_df = test_calc_df = None
test_df.head(1).T
```
<!-- nb-output hash="2bd7d2a4359cf9f7" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><div>
<style>.dataframe tbody tr th:only-of-type { vertical-align: middle; } .dataframe tbody tr th { vertical-align: top; } .dataframe thead th { text-align: right; }</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>patient_id</th>
      <td>P_00016</td>
    </tr>
    <tr>
      <th>breast_density</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>left or right breast</th>
      <td>LEFT</td>
    </tr>
    <tr>
      <th>image view</th>
      <td>CC</td>
    </tr>
    <tr>
      <th>abnormality id</th>
      <td>1</td>
    </tr>
    <tr>
      <th>abnormality type</th>
      <td>mass</td>
    </tr>
    <tr>
      <th>mass shape</th>
      <td>IRREGULAR</td>
    </tr>
    <tr>
      <th>mass margins</th>
      <td>SPICULATED</td>
    </tr>
    <tr>
      <th>assessment</th>
      <td>5</td>
    </tr>
    <tr>
      <th>pathology</th>
      <td>MALIGNANT</td>
    </tr>
    <tr>
      <th>subtlety</th>
      <td>5</td>
    </tr>
    <tr>
      <th>image file path</th>
      <td>Mass-Test_P_00016_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.416403281812750683720028031170500130104/1.3.6.1.4.1.9590.100.1.2.245063149211255120613007755642780114172/1-1.dcm</td>
    </tr>
    <tr>
      <th>cropped image file path</th>
      <td>Mass-Test_P_00016_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.259596319110047779433501728143778409887/1.3.6.1.4.1.9590.100.1.2.30820586311062570442302321942433426184/1-2.dcm</td>
    </tr>
    <tr>
      <th>ROI mask file path</th>
      <td>Mass-Test_P_00016_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.259596319110047779433501728143778409887/1.3.6.1.4.1.9590.100.1.2.30820586311062570442302321942433426184/1-1.dcm</td>
    </tr>
    <tr>
      <th>breast density</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>calc type</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>calc distribution</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div></div>
</div>
<!-- /nb-output -->

Then combine both splits for analysis:

```python
all_df = pd.concat([train_df, test_df])
```
<!-- nb-output hash="ded601e62f016f62" format="html" -->

<!-- /nb-output -->

The `metadata.csv` file maps the CSV path references to actual file locations on disk:

```python
metadata_df = pd.read_csv(DATASET_ROOT / "metadata.csv")
metadata_df.head(1).T
```
<!-- nb-output hash="3a75cfa83be89783" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><div>
<style>.dataframe tbody tr th:only-of-type { vertical-align: middle; } .dataframe tbody tr th { vertical-align: top; } .dataframe thead th { text-align: right; }</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Series UID</th>
      <td>1.3.6.1.4.1.9590.100.1.2.419081637812053404913157930753972718515</td>
    </tr>
    <tr>
      <th>Collection</th>
      <td>CBIS-DDSM</td>
    </tr>
    <tr>
      <th>3rd Party Analysis</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Data Description URI</th>
      <td>https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY</td>
    </tr>
    <tr>
      <th>Subject ID</th>
      <td>Calc-Test_P_00038_LEFT_CC_1</td>
    </tr>
    <tr>
      <th>Study UID</th>
      <td>1.3.6.1.4.1.9590.100.1.2.161465562211359959230647609981488894942</td>
    </tr>
    <tr>
      <th>Study Description</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Study Date</th>
      <td>08-29-2017</td>
    </tr>
    <tr>
      <th>Series Description</th>
      <td>ROI mask images</td>
    </tr>
    <tr>
      <th>Manufacturer</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Modality</th>
      <td>MG</td>
    </tr>
    <tr>
      <th>SOP Class Name</th>
      <td>Secondary Capture Image Storage</td>
    </tr>
    <tr>
      <th>SOP Class UID</th>
      <td>1.2.840.10008.5.1.4.1.1.7</td>
    </tr>
    <tr>
      <th>Number of Images</th>
      <td>2</td>
    </tr>
    <tr>
      <th>File Size</th>
      <td>14.06 MB</td>
    </tr>
    <tr>
      <th>File Location</th>
      <td>./CBIS-DDSM/Calc-Test_P_00038_LEFT_CC_1/08-29-2017-DDSM-NA-94942/1.000000-ROI mask images-18515</td>
    </tr>
    <tr>
      <th>Download Timestamp</th>
      <td>2025-11-25T05:30:27.736</td>
    </tr>
  </tbody>
</table>
</div></div>
</div>
<!-- /nb-output -->

CBIS-DDSM contains 1,566 studies.

```python
images_per_patient = all_df.groupby("patient_id").size()
print(f"Total patients: {len(images_per_patient)}")
```
<!-- nb-output hash="0080b99af4382486" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">Total patients: 1566
</pre>
</div>
<!-- /nb-output -->

They separate the studies by the abnormality type present, either "mass" or "calcification".

The paper[^1] claims there are 891 mass cases, although the actual dataset appears to have 892 mass abnormalities.

```python
len(all_df[all_df["abnormality type"] == "mass"].patient_id.unique())
```
<!-- nb-output hash="955c85dbc448c46f" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">892</pre>
</div>
<!-- /nb-output -->

The paper also describes 753 calcification abnormalities that match what we see.

```python
len(all_df[all_df["abnormality type"] == "calcification"].patient_id.unique())
```
<!-- nb-output hash="595969226c6a7286" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">753</pre>
</div>
<!-- /nb-output -->

We know that a mammogram consists of 2 images per breast: a craniocaudal (CC) view from above and a mediolateral oblique (MLO) view from the side.

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
<!-- nb-output id="cbis-ddsm-images-per-patient" hash="d6d2093c8466f366" format="image" -->
![Bar chart of images per patient, with 1,005 patients having two images and 336 having one.](../_media/cbis-ddsm-images-per-patient.png)
<!-- /nb-output -->

Looking at the distribution of images per patient, about 1005 patients have both views, while many have only a single image, which is basically an incomplete mammogram (containing only one view per patient, instead of the expected two).

## Fetching Images {id="fetching images"}

Even with the corrected CSVs, we still need to do a few things to look up the DICOM images for each study.

The file paths in the CSV (e.g. `Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/1-1.dcm`) aren't direct paths to the files on disk. They encode metadata: subject ID, study UID, series UID, and filename. We need to parse these components and cross-reference with `metadata.csv` to find the actual file location.

```python
from pydantic import BaseModel

class DCMData(BaseModel):
    subject_id: str
    study_uid: str
    series_uid: str
    dcm_file: str

def get_file_data_from_dcm(dcm_path: str) -> DCMData:
    """Parse DICOM path string to extract metadata components."""
    data = str(dcm_path).strip().split("/")
    dcm_og = data[-1].strip().split(".")[0]
    return DCMData(
        subject_id=data[0],
        study_uid=data[1],
        series_uid=data[2],
        dcm_file=dcm_og,
    )
```
<!-- nb-output hash="b69fb3c06b725d2a" format="html" -->

<!-- /nb-output -->

```python
def get_filepath_from_dcm_data(dcm_data: DCMData) -> Path:
    """Look up actual file path from metadata using DCM data."""
    meta_row = metadata_df[
        (metadata_df["Subject ID"] == dcm_data.subject_id)
        & (metadata_df["Series UID"] == dcm_data.series_uid)
        & (metadata_df["Study UID"] == dcm_data.study_uid)
    ].iloc[0]
    file_location = meta_row["File Location"]
    return DATASET_ROOT / Path(file_location) / (dcm_data.dcm_file + ".dcm")
```
<!-- nb-output hash="d45ef2e574796d15" format="html" -->

<!-- /nb-output -->

Now we can load the DICOM images using pydicom:

```python
def dicom_to_array(file_path: Path):
    """Load a DICOM file and return pixel array."""
    ds = pydicom.dcmread(file_path)
    return ds.pixel_array
```
<!-- nb-output hash="074a1dbc7f9bcd3c" format="html" -->

<!-- /nb-output -->

Let's load an example patient to see all three image types (full mammogram, ROI mask, and cropped region):

```python
patient_df = all_df[all_df.patient_id == "P_00065"]
row = patient_df[patient_df["image view"] == "CC"].iloc[0]

img_path = get_filepath_from_dcm_data(get_file_data_from_dcm(row["image file path"]))
mask_path = get_filepath_from_dcm_data(get_file_data_from_dcm(row["ROI mask file path"]))
crop_path = get_filepath_from_dcm_data(get_file_data_from_dcm(row["cropped image file path"]))

original_img = dicom_to_array(img_path)
mask_img = dicom_to_array(mask_path)
crop_img = dicom_to_array(crop_path)
```
<!-- nb-output hash="51e22999140779a4" format="html" -->

<!-- /nb-output -->

Visualising the original mammogram, the binary ROI mask, an overlay of the two, and the cropped abnormality region:

```python
fig, axes = plt.subplots(1, 4, figsize=(12, 5))

axes[0].imshow(original_img, cmap="gray")
axes[0].axis("off")

axes[1].imshow(mask_img, cmap="gray")
axes[1].axis("off")

axes[2].imshow(original_img, cmap="gray")
axes[2].imshow(mask_img, cmap="jet", alpha=0.4)
axes[2].axis("off")

axes[3].imshow(crop_img, cmap="gray")
axes[3].axis("off")

plt.tight_layout()
fig.subplots_adjust(top=0.92, wspace=0.01)

titles = ["Original", "ROI Mask", "Overlay", "Cropped ROI"]
for ax, title in zip(axes, titles):
    x = ax.get_position().x0 + ax.get_position().width / 2
    fig.text(x, 0.98, title, ha="center", va="top", fontsize=12, fontweight="bold")
plt.show()
```
<!-- nb-output id="cbis-ddsm-patient-image-types" hash="2861758c85b8af7d" format="image" -->
![Four panels showing a mammogram, its region-of-interest mask, the mask overlaid on the mammogram, and the cropped region.](../_media/cbis-ddsm-patient-image-types.png)
<!-- /nb-output -->

## Key Metadata {id="key metadata"}

The most important label is `pathology`, indicating whether the abnormality is **benign**, **benign_without_callback** (clearly no risk of malignancy), or **malignant**.

```python
pathology_counts = all_df['pathology'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(pathology_counts.index, pathology_counts.values, color=['#2ecc71', '#3498db', '#e74c3c'])
plt.title('Pathology Distribution', fontsize=12, fontweight='bold')
plt.ylabel('Count')
plt.xticks(rotation=45)
for i, v in enumerate(pathology_counts.values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()
```
<!-- nb-output id="cbis-ddsm-pathology-distribution" hash="8b2bd919af56e3e3" format="image" -->
![Bar chart showing 1,457 malignant, 1,429 benign and 682 benign-without-callback abnormalities.](../_media/cbis-ddsm-pathology-distribution.png)
<!-- /nb-output -->

The dataset also includes [Breast Imaging Reporting and Data System (BI-RADS)](breast-imaging-reporting-and-data-system-bi-rads.md) assessment categories (0-6) that indicate the level of suspicion. The distribution shows most cases fall into categories 4 and 5 (suspicious/highly suggestive of malignancy), which makes sense given that this is a dataset specifically curated around abnormalities.

```python
assessment_counts = all_df['assessment'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(assessment_counts.index.astype(str), assessment_counts.values, color='#9b59b6')
plt.title('BI-RADS Assessment Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Assessment Category')
plt.ylabel('Count')
for i, (idx, v) in enumerate(assessment_counts.items()):
    plt.text(i, v + 5, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()
```
<!-- nb-output id="cbis-ddsm-assessment-distribution" hash="259b68252dd619bf" format="image" -->
![Bar chart of BI-RADS assessments from 0 to 5, with category 4 the largest group at 1,633 cases.](../_media/cbis-ddsm-assessment-distribution.png)
<!-- /nb-output -->

### Mass Descriptors {id="mass descriptors"}

For mass abnormalities, the dataset includes shape and margin descriptors - clinically important features for diagnosis:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

shape_counts = all_df['mass shape'].dropna().value_counts()
axes[0].barh(shape_counts.index, shape_counts.values, color='#e67e22')
axes[0].set_title('Mass Shape Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Count')

margins_counts = all_df['mass margins'].dropna().value_counts()
axes[1].barh(margins_counts.index, margins_counts.values, color='#16a085')
axes[1].set_title('Mass Margins Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')

plt.tight_layout()
plt.show()
```
<!-- nb-output id="cbis-ddsm-mass-descriptors" hash="78f572c3a706c20f" format="image" -->
![Horizontal bar charts comparing recorded mass shapes and mass margins, including combined descriptors.](../_media/cbis-ddsm-mass-descriptors.png)
<!-- /nb-output -->

Irregular shapes and spiculated margins are typically more concerning for malignancy, while oval/round shapes with circumscribed margins tend to be benign.

### Breast Density {id="breast density"}

Breast density (1-4 scale) affects mammogram interpretation - denser tissue makes abnormalities harder to detect:

```python
density_col = 'breast_density' if 'breast_density' in all_df.columns else 'breast density'
breast_density_counts = all_df[density_col].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(breast_density_counts.index.astype(str), breast_density_counts.values, color='#2980b9')
plt.title('Breast Density Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Breast Density (1=fatty, 4=extremely dense)')
plt.ylabel('Count')
for i, v in enumerate(breast_density_counts.values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()
```
<!-- nb-output id="cbis-ddsm-breast-density-distribution" hash="bdddf40fc24e502b" format="image" -->
![Bar chart of breast-density categories 1 to 4, with category 2 the most common at 757 cases.](../_media/cbis-ddsm-breast-density-distribution.png)
<!-- /nb-output -->

## Train/Test Split {id="train/test split"}

The authors provide standardised train/test splits (80/20), stratified by BI-RADS assessment to ensure similar difficulty distribution:

| Set | Benign Cases | Malignant Cases |
|-----|--------------|-----------------|
| Calcification Training | 329 (552 abnormalities) | 273 (304 abnormalities) |
| Calcification Test | 85 (112 abnormalities) | 66 (77 abnormalities) |
| Mass Training | 355 (387 abnormalities) | 336 (361 abnormalities) |
| Mass Test | 117 (135 abnormalities) | 83 (87 abnormalities) |

Note that case counts differ from abnormality counts since some cases contain multiple abnormalities.

## Segmentation Quality {id="segmentation quality"}

The authors validated their Chan-Vese segmentations against hand-drawn ROIs from an experienced radiologist on 118 images. The mean Dice coefficient between computer-generated and hand-drawn ROIs was **0.792 ± 0.108**, compared to **0.398 ± 0.195** for the original DDSM annotations vs hand-drawn. These results represent a statistically significant improvement (Wilcoxon signed-rank test, p = 5.54 × 10⁻¹⁹).

During curation, 339 mass images where the lesion was not clearly visible were removed after review by a trained mammographer.

## Limitations {id="limitations"}

While CBIS-DDSM is valuable for research, it has some limitations worth noting. The original DDSM images were digitised from film mammograms, not acquired digitally. Modern Full-Field Digital Mammography (FFDM) systems produce higher-quality images, and newer datasets like [InBreast](inbreast.md) and [VinDr-Mammo](vindr-mammo.md) tend to contain these sorts of images. Additionally, DDSM images are focused on a specific abnormality, but a breast may contain multiple abnormalities, warranting investigation. Lastly, the original DDSM was collected in the 1990s, so imaging quality and patient demographics may differ from those in contemporary datasets.

Despite these limitations, CBIS-DDSM remains one of the most widely used public mammography datasets for developing and benchmarking CAD algorithms.

## References {id="references"}

[^1]: Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*, 4(1), 170177. [https://doi.org/10.1038/sdata.2017.177](https://doi.org/10.1038/sdata.2017.177)
[^2]: Heath, M., Bowyer, K., Kopans, D., Moore, R. & Kegelmeyer, W. P. The Digital Database for Screening Mammography. Proceedings of the Fifth International Workshop on Digital Mammography 212–218 (2001). Available at http://marathon.csee.usf.edu/Mammography/software/HeathEtAlIWDM_2000.pdf
[^3]: Suckling, J. et al. The Mammographic Image Analysis Society digital mammogram database. *Exerpta Medica* 375–378 (1994). [http://peipa.essex.ac.uk/info/mias.html](http://peipa.essex.ac.uk/info/mias.html)
[^4]: Lehmann, T. M. et al. IRMA—Content-based image retrieval in medical applications. *Methods Inf. Med.* 43, 354–361 (2004).
