"""
Feature extraction: builds the two feature tables the models are trained on.

    data/baseline_features.csv   asymmetry, border and HSV colour features of the original images
    data/extended_features.csv   the same after hair/pen removal, plus RGB and greyscale features

Needs the PAD-UFES-20 images in data/imgs/ and the lesion masks in data/masks/. Run from the
project root:  python src/main_extraction.py
"""
import os

import cv2
import pandas as pd

from feature_A import get_asymmetry
from feature_B import get_compactness
from feature_C import get_color_data, get_rgb_data
from shortcuts import apply_shortcuts, load_shortcut_annotations
from utils import (
    BASELINE_FEATURE_COLUMNS, BASELINE_FEATURES_CSV_PATH, EXTENDED_FEATURE_COLUMNS, EXTENDED_FEATURES_CSV_PATH,
    IMAGE_EXTENSIONS, IMGS_DIR, MASK_DIR, METADATA_CSV_PATH, normalize_img_id,
)


def find_file(directory, name):
    """Path of directory/name with a .png/.jpg/.jpeg extension, or None if there is none."""
    for extension in IMAGE_EXTENSIONS:
        path = os.path.join(directory, name + extension)
        if os.path.exists(path):
            return path
    return None


def load_image_and_mask(img_id):
    """Load an image as RGB and its lesion mask as greyscale.

    Raises an error if either file is missing or unreadable, if the mask is empty or if
    the image and mask sizes differ.
    """
    image_path = find_file(IMGS_DIR, img_id)
    mask_path = find_file(MASK_DIR, img_id + "_mask")
    if image_path is None or mask_path is None:
        raise FileNotFoundError(f"No image or mask found for {img_id}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # OpenCV loads images as BGR
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"Unreadable image or mask for {img_id}")
    if not (mask > 0).any():
        raise ValueError(f"Empty mask for {img_id}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image and mask sizes differ for {img_id}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask


def is_usable(img_id):
    """True if the image and its mask load without problems."""
    try:
        load_image_and_mask(img_id)
        return True
    except (FileNotFoundError, ValueError):
        return False


def load_dataset_rows():
    """patient_id, diagnostic and img_id of every metadata row with a usable image and mask
    (2095 of the 2298 rows), sorted by img_id."""
    rows = pd.read_csv(METADATA_CSV_PATH)[["patient_id", "diagnostic", "img_id"]].copy()
    rows["img_id"] = rows["img_id"].apply(normalize_img_id)
    rows = rows[rows["img_id"].apply(is_usable)]
    return rows.sort_values("img_id").reset_index(drop=True)


def extract_features_for_row(row, annotations=None, include_rgb=False):
    """Features of one image.

    The shape features (A, B) always come from the mask. When annotations are given, hair
    and pen marks are removed from the image before the colour features (C) are computed.
    """
    image, mask = load_image_and_mask(row["img_id"])
    if annotations is not None:
        image, _, _ = apply_shortcuts(image, row["img_id"], annotations)

    features = {
        "patient_id": row["patient_id"],
        "diagnostic": row["diagnostic"],
        "img_id": row["img_id"],
        "asymmetry_score": get_asymmetry(mask),
        "border_score": get_compactness(mask),
    }
    features.update(get_color_data(image, mask))
    if include_rgb:
        features.update(get_rgb_data(image, mask))
    return features


def extract_features(output_path, columns, annotations=None, include_rgb=False):
    """Extract the features of every usable image and save them as a CSV table."""
    rows = load_dataset_rows()
    results = []
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        results.append(extract_features_for_row(row, annotations, include_rgb))
        if i % 50 == 0:
            print(f"Processed {i}/{len(rows)} rows", flush=True)

    features = pd.DataFrame(results).reindex(columns=columns)
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} feature rows to {output_path}")
    return features


if __name__ == "__main__":
    extract_features(BASELINE_FEATURES_CSV_PATH, BASELINE_FEATURE_COLUMNS)
    extract_features(EXTENDED_FEATURES_CSV_PATH, EXTENDED_FEATURE_COLUMNS,
                     annotations=load_shortcut_annotations(), include_rgb=True)
