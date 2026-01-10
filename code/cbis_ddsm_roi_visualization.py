"""
Generate a 4-panel visualization of CBIS-DDSM mammography data showing:
1. Original mammogram
2. ROI mask
3. Original with mask overlay
4. Cropped ROI

Usage:
    python cbis_ddsm_roi_visualization.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from pydantic import BaseModel


DATASET_ROOT = Path.home() / "datasets" / "CBIS-DDSM"
OUTPUT_PATH = Path(__file__).parent.parent / "notes" / "_media" / "cbis-ddsm-patient-assets.png"


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


def get_filepath_from_dcm_data(dcm_data: DCMData, metadata_df: pd.DataFrame) -> Path:
    """Look up actual file path from metadata using DCM data."""
    meta_row = metadata_df[
        (metadata_df["Subject ID"] == dcm_data.subject_id)
        & (metadata_df["Series UID"] == dcm_data.series_uid)
        & (metadata_df["Study UID"] == dcm_data.study_uid)
    ].iloc[0]
    file_location = meta_row["File Location"]
    return DATASET_ROOT / Path(file_location) / (dcm_data.dcm_file + ".dcm")


def dicom_to_array(file_path: Path) -> np.ndarray:
    """Load a DICOM file and return pixel array."""
    ds = pydicom.dcmread(file_path)
    return ds.pixel_array


def get_patient_images(row: pd.Series, metadata_df: pd.DataFrame) -> tuple[Path, Path, Path]:
    """Get paths to original image, mask, and cropped ROI for a patient row."""
    img_path = get_filepath_from_dcm_data(
        get_file_data_from_dcm(row["image file path"]), metadata_df
    )
    mask_path = get_filepath_from_dcm_data(
        get_file_data_from_dcm(row["ROI mask file path"]), metadata_df
    )
    crop_path = get_filepath_from_dcm_data(
        get_file_data_from_dcm(row["cropped image file path"]), metadata_df
    )
    return img_path, mask_path, crop_path


def create_roi_visualization(
    patient_id: str = "P_00065",
    view: str = "CC",
    output_path: Path = OUTPUT_PATH,
):
    """Create 4-panel visualization for a given patient and view."""
    # Load metadata
    train_mass_df = pd.read_csv(DATASET_ROOT / "mass_case_description_train_set.csv")
    train_calc_df = pd.read_csv(DATASET_ROOT / "calc_case_description_train_set.csv")
    test_mass_df = pd.read_csv(DATASET_ROOT / "mass_case_description_test_set.csv")
    test_calc_df = pd.read_csv(DATASET_ROOT / "calc_case_description_test_set.csv")
    all_df = pd.concat([train_mass_df, train_calc_df, test_mass_df, test_calc_df])
    metadata_df = pd.read_csv(DATASET_ROOT / "metadata.csv")

    # Get patient data
    patient_df = all_df[all_df.patient_id == patient_id]
    patient_row = patient_df[patient_df["image view"] == view].iloc[0]

    # Get image paths
    img_path, mask_path, crop_path = get_patient_images(patient_row, metadata_df)

    # Load images
    original_img = dicom_to_array(img_path)
    mask_img = dicom_to_array(mask_path)
    crop_img = dicom_to_array(crop_path)

    # Create figure with consistent title alignment
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))

    # Panel 1: Original
    axes[0].imshow(original_img, cmap="gray")
    axes[0].axis("off")

    # Panel 2: Mask
    axes[1].imshow(mask_img, cmap="gray")
    axes[1].axis("off")

    # Panel 3: Original with mask overlay
    axes[2].imshow(original_img, cmap="gray")
    axes[2].imshow(mask_img, cmap="jet", alpha=0.4)
    axes[2].axis("off")

    # Panel 4: Cropped ROI
    axes[3].imshow(crop_img, cmap="gray")
    axes[3].axis("off")

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, wspace=0.01)

    # Add titles at consistent y position using figure coordinates
    titles = ["Original", "ROI Mask", "Overlay", "Cropped ROI"]
    for ax, title in zip(axes, titles):
        x = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x, 0.98, title, ha="center", va="top", fontsize=12, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    create_roi_visualization()
